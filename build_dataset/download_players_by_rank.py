import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import time
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import urllib.request

import aiofiles
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bp_core.dataset_versions import resolve_target_dataset_version  # noqa: E402
from bp_core.rank_tiers import (  # noqa: E402
    MAX_DEFAULT_RANK_TO_SCAN,
    RANK_TIER_SPECS,
    RANK_TIER_TARGET_PLAYERS,
    enrich_rank_context,
    tier_targets_met,
)

RANKING_URL = "https://lucksack.gg/players"
DETAIL_URL = "https://lucksack.gg/player/{}"

DATASET_ROOT = Path("./dataset")
SAVE_DIR = DATASET_ROOT
CSV_PATH = SAVE_DIR / "ranking_list.csv"
FETCH_STATE_PATH = Path("./data/player_fetch_state.json")

PAGE_TIMEOUT = 30000
DETAIL_WAIT_MS = 3000
DETAIL_READY_POLL_MS = 250
LEADERBOARD_SETTLE_MS = 300
LEADERBOARD_CLICK_SETTLE_MS = 150
CONCURRENCY = 2
MAX_PAGES = 160
MAX_RANK_TO_SCAN = MAX_DEFAULT_RANK_TO_SCAN
TARGET_PLAYERS_PER_TIER = dict(RANK_TIER_TARGET_PLAYERS)
OTHER_TIER_MIN_RANK = next(spec["rank_min"] for spec in RANK_TIER_SPECS if spec["key"] == "other")
HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "1") != "0"
BROWSER_PROFILE_DIR = Path(os.getenv("BP_BROWSER_PROFILE_DIR", "./.playwright_profile/lucksack"))
USE_EXISTING_VERSION_CSV = os.getenv("BP_USE_EXISTING_VERSION_CSV", "0") == "1"
CHALLENGE_TIMEOUT_MS = int(os.getenv("BP_CF_AUTO_WAIT_MS", "60000"))
MANUAL_CHALLENGE_TIMEOUT_MS = int(os.getenv("BP_CF_MANUAL_WAIT_MS", "300000"))
ENABLE_MANUAL_CHALLENGE = os.getenv("BP_CF_ALLOW_MANUAL", "1") != "0"

ROW_SELECTOR = 'tbody[data-slot="table-body"] tr[data-slot="table-row"]'
NEXT_SELECTOR = 'button:has-text("Next")'
PLAYER_LINK_SELECTOR = 'tbody[data-slot="table-body"] a[href^="/player/"]'
CHALLENGE_TEXTS = (
    "Performing security verification",
    "Just a moment...",
    "Enable JavaScript and cookies to continue",
)
MAINTENANCE_TEXTS = (
    "scheduled maintenance",
    "the site may be unavailable at times",
)
EMPTY_LEADERBOARD_TEXTS = (
    "no players found for this selection",
)
MATCH_HISTORY_READY_PATTERNS = (
    "victory",
    "defeat",
)
DETAIL_MIN_INTERVAL_SECONDS = float(os.getenv("BP_DETAIL_MIN_INTERVAL_SECONDS", "1.5"))
DETAIL_MAX_INTERVAL_SECONDS = float(os.getenv("BP_DETAIL_MAX_INTERVAL_SECONDS", "2.5"))
MAX_DETAIL_FETCHES_PER_RUN = int(os.getenv("BP_MAX_DETAIL_FETCHES_PER_RUN", "180"))
ACTIVE_REVISIT_HOURS = float(os.getenv("BP_ACTIVE_REVISIT_HOURS", "24"))
DEFAULT_REVISIT_HOURS = float(os.getenv("BP_DEFAULT_REVISIT_HOURS", "72"))
COOLDOWN_REVISIT_HOURS = float(os.getenv("BP_COOLDOWN_REVISIT_HOURS", "168"))
STALE_REVISIT_DAYS = float(os.getenv("BP_STALE_REVISIT_DAYS", "30"))


def parse_args():
    parser = argparse.ArgumentParser(description="Download ranked player BP pages into a versioned dataset directory.")
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Target dataset version, for example v2.0. Defaults to BP_DATASET_VERSION or the next auto-incremented version.",
    )
    parser.add_argument(
        "--use-existing-version-csv",
        action="store_true",
        help="Reuse dataset/<version>/ranking_list.csv for resume. Disabled by default so each new version crawls fresh rankings.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Attach to a Chrome window you already opened, instead of launching a browser.",
    )
    parser.add_argument(
        "--cdp-url",
        default="http://127.0.0.1:9333",
        help="DevTools address of the Chrome window used with --headful.",
    )
    parser.add_argument(
        "--challenge-timeout-ms",
        type=int,
        default=CHALLENGE_TIMEOUT_MS,
        help="Automatic wait time for Cloudflare verification before falling back to manual solve.",
    )
    parser.add_argument(
        "--manual-challenge-timeout-ms",
        type=int,
        default=MANUAL_CHALLENGE_TIMEOUT_MS,
        help="Extra wait time for manually solving Cloudflare in headful mode.",
    )
    parser.add_argument(
        "--disable-manual-challenge",
        action="store_true",
        help="Disable manual Cloudflare fallback even in headful mode.",
    )
    parser.add_argument(
        "--detail-concurrency",
        type=int,
        default=CONCURRENCY,
        help="Detail-page concurrency. Keep low to reduce impact on the site.",
    )
    parser.add_argument(
        "--max-detail-fetches",
        type=int,
        default=MAX_DETAIL_FETCHES_PER_RUN,
        help="Maximum number of player detail pages to fetch in one run.",
    )
    parser.add_argument(
        "--detail-min-interval-seconds",
        type=float,
        default=DETAIL_MIN_INTERVAL_SECONDS,
        help="Minimum delay between detail page requests per worker.",
    )
    parser.add_argument(
        "--detail-max-interval-seconds",
        type=float,
        default=DETAIL_MAX_INTERVAL_SECONDS,
        help="Maximum delay between detail page requests per worker.",
    )
    return parser.parse_args()


def configure_output_paths(dataset_version):
    global SAVE_DIR, CSV_PATH
    SAVE_DIR = DATASET_ROOT / dataset_version
    CSV_PATH = SAVE_DIR / "ranking_list.csv"
    return SAVE_DIR


async def build_context(browser):
    return await browser.new_context(
        viewport={"width": 1440, "height": 900},
        locale="en-US",
    )


async def build_persistent_context(playwright):
    profile_dir = BROWSER_PROFILE_DIR.expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] using persistent browser profile: {profile_dir.resolve()}")
    # Bundled Chromium is often reset by the site before a page exists.
    # The installed Chrome keeps one verification across later navigations.
    return await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        channel="chrome",
        headless=False,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
    )


def _connection_closed(exc):
    return "ERR_CONNECTION_CLOSED" in str(exc) or "ERR_CONNECTION_RESET" in str(exc)


async def goto_with_retry(page, url, timeout=PAGE_TIMEOUT, attempts=4):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception as exc:
            last_error = exc
            if not _connection_closed(exc) or attempt == attempts:
                raise
            wait_seconds = 2 * attempt
            print(f"[WARN] connection closed for {url}; retry {attempt}/{attempts - 1} in {wait_seconds}s")
            await page.wait_for_timeout(wait_seconds * 1000)
    raise last_error


async def setup_page(page):
    # Leave images and scripts alone. Aborting them reloads the Cloudflare
    # widget on the next navigation, so a solved check does not stick.
    return None


async def save_debug(page, name):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        html = await page.content()
        (SAVE_DIR / f"{name}.html").write_text(html, encoding="utf-8")
    except Exception:
        pass
    try:
        await page.screenshot(path=str(SAVE_DIR / f"{name}.png"), full_page=True)
    except Exception:
        pass


async def is_challenge_page(page):
    """Detect the top-level Cloudflare page only.

    Hidden challenge frames stay in the document after a successful check.
    Searching every frame made the script treat a cleared page as a new
    challenge and ask for verification again.
    """
    try:
        title = (await page.title()).strip().lower()
    except Exception:
        title = ""

    if title in {"just a moment...", "attention required!"}:
        return True
    if "__cf_chl_" in page.url or "challenge-platform" in page.url:
        return True

    try:
        body = (await page.locator("body").inner_text(timeout=2000)).lower()
    except Exception:
        return False
    return any(text.lower() in body for text in CHALLENGE_TEXTS) or "performing security verification" in body


async def wait_for_challenge_clear(page, timeout_ms=45000):
    elapsed = 0
    step = 2000
    warned = False

    while elapsed < timeout_ms:
        if not await is_challenge_page(page):
            return True

        if not warned:
            print("[INFO] waiting for Cloudflare verification page to clear")
            warned = True

        await page.wait_for_timeout(step)
        elapsed += step

    return False


async def ensure_challenge_cleared(page, context_label, auto_timeout_ms=None, manual_timeout_ms=None):
    auto_timeout_ms = CHALLENGE_TIMEOUT_MS if auto_timeout_ms is None else max(0, int(auto_timeout_ms))
    manual_timeout_ms = MANUAL_CHALLENGE_TIMEOUT_MS if manual_timeout_ms is None else max(0, int(manual_timeout_ms))

    if not await is_challenge_page(page):
        return True

    cleared = await wait_for_challenge_clear(page, timeout_ms=auto_timeout_ms)
    if cleared:
        return True

    await save_debug(page, f"debug_{context_label}_cloudflare")

    if HEADLESS or not ENABLE_MANUAL_CHALLENGE:
        if HEADLESS:
            print(
                "[WARN] Cloudflare challenge did not clear automatically in headless mode. "
                "Re-run with --headful or set PLAYWRIGHT_HEADLESS=0."
            )
        return False

    print(
        "[ACTION] Cloudflare challenge needs manual solve. "
        f"Browser will stay open for up to {manual_timeout_ms // 1000} seconds. "
        "Complete the verification in the browser, then the script will continue automatically."
    )
    return await wait_for_challenge_clear(page, timeout_ms=manual_timeout_ms)


async def wait_for_rows(page, timeout_ms=30000):
    elapsed = 0
    step = 250

    while elapsed < timeout_ms:
        count = await page.locator(ROW_SELECTOR).count()
        if count > 0:
            return count

        try:
            await page.mouse.wheel(0, 1500)
        except Exception:
            pass

        await page.wait_for_timeout(step)
        elapsed += step

    return 0


async def wait_for_leaderboard_ready(page, timeout_ms=45000):
    elapsed = 0
    step = 1500
    saw_loading = False

    while elapsed < timeout_ms:
        if await is_challenge_page(page):
            cleared = await wait_for_challenge_clear(page, timeout_ms=min(15000, timeout_ms - elapsed))
            if not cleared:
                return False

        player_link_count = await page.locator(PLAYER_LINK_SELECTOR).count()
        if player_link_count > 0:
            return True

        page_text = ""
        try:
            page_text = await page.locator("body").inner_text(timeout=3000)
        except Exception:
            pass

        if "Loading..." in page_text:
            saw_loading = True

        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        try:
            await page.mouse.wheel(0, 1200)
        except Exception:
            pass

        await page.wait_for_timeout(step)
        elapsed += step

    return not saw_loading and await page.locator(PLAYER_LINK_SELECTOR).count() > 0


def leaderboard_unavailable_reason(page_text):
    text = " ".join(str(page_text or "").lower().split())
    if any(marker in text for marker in MAINTENANCE_TEXTS):
        return "Lucksack 正在计划维护，排行榜暂时不可用"
    if re.search(r"(?<!\d)0 players found\b", text) or any(
        marker in text for marker in EMPTY_LEADERBOARD_TEXTS
    ):
        return "Lucksack 排行榜明确返回 0 名玩家"
    return ""


async def page_text(page):
    try:
        return await page.locator("body").inner_text(timeout=3000)
    except Exception:
        return ""


async def extract_players_from_page(page):
    js = f"""
    () => {{
        const rows = Array.from(document.querySelectorAll('{ROW_SELECTOR}'));
        const results = [];

        for (const row of rows) {{
            const tds = row.querySelectorAll('td');
            if (tds.length < 2) continue;

            let rank = null;
            const rankText = (tds[0].textContent || '').trim();
            const rankMatch = rankText.match(/\\d+/);
            if (rankMatch) {{
                rank = parseInt(rankMatch[0], 10);
            }}

            const a = row.querySelector('a[href^="/player/"]');
            if (!a) continue;

            const href = a.getAttribute('href') || '';
            const idMatch = href.match(/^\\/player\\/(\\d+)$/);
            if (!idMatch) continue;

            const player_id = idMatch[1];

            let player_name = '';
            const spans = a.querySelectorAll('span');
            if (spans.length > 0) {{
                player_name = spans[spans.length - 1].textContent.trim();
            }}
            if (!player_name) {{
                player_name = (a.textContent || '').trim();
            }}

            results.push({{
                rank,
                player_id,
                player_name,
                href
            }});
        }}

        return results;
    }}
    """
    return await page.evaluate(js)


def annotate_player(player):
    info = enrich_rank_context(rank=player.get("rank"))
    item = dict(player)
    item.update(info)
    return item


def annotate_players(players):
    return [annotate_player(player) for player in players]


def format_tier_progress(counter):
    parts = []
    for spec in RANK_TIER_SPECS:
        key = spec["key"]
        label = spec["label"]
        target = TARGET_PLAYERS_PER_TIER.get(key)
        current = counter.get(key, 0)
        if target is None:
            parts.append(f"{label}:{current}")
        else:
            parts.append(f"{label}:{current}/{target}")
    return " | ".join(parts)


def is_player_selectable(player, selected_ids, tier_counter):
    player_id = player["player_id"]
    if player_id in selected_ids:
        return False

    tier_key = player.get("rank_tier")
    target = TARGET_PLAYERS_PER_TIER.get(tier_key)
    if target is not None and tier_counter.get(tier_key, 0) >= int(target):
        return False

    return True


def get_selectable_players(players, selected_ids, tier_counter):
    return [player for player in players if is_player_selectable(player, selected_ids, tier_counter)]


def summarize_page_window(players):
    ranks = [player.get("rank") for player in players if isinstance(player.get("rank"), int)]
    tiers = []
    seen = set()
    for player in players:
        tier_key = player.get("rank_tier")
        if tier_key and tier_key not in seen:
            seen.add(tier_key)
            tiers.append(tier_key)

    return {
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "tiers": tiers,
    }


def html_has_ready_match_history(html: str) -> bool:
    if not html:
        return False

    lower_html = html.lower()
    if "match history" not in lower_html:
        return False

    monster_link_count = html.count("/monster/")
    if monster_link_count < 10:
        return False

    return any(pattern in lower_html for pattern in MATCH_HISTORY_READY_PATTERNS)


def saved_player_page_is_ready(path: Path) -> bool:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return html_has_ready_match_history(html)


async def wait_for_match_history(page, timeout_ms=DETAIL_WAIT_MS):
    """Return as soon as the rendered match history is present."""
    timeout_ms = max(0, int(timeout_ms))
    elapsed = 0
    while elapsed < timeout_ms:
        html = await page.content()
        if html_has_ready_match_history(html):
            return html
        wait_ms = min(DETAIL_READY_POLL_MS, timeout_ms - elapsed)
        await page.wait_for_timeout(wait_ms)
        elapsed += wait_ms
    return await page.content()


def normalize_and_sort(players):
    merged = {}
    for p in players:
        pid = p["player_id"]
        rank = p.get("rank")
        if pid not in merged:
            merged[pid] = p
        else:
            old_rank = merged[pid].get("rank")
            if old_rank is None and rank is not None:
                merged[pid] = p
            elif old_rank is not None and rank is not None and rank < old_rank:
                merged[pid] = p

    result = list(merged.values())
    result.sort(key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 10**9, int(x["player_id"])))
    return result


def parse_optional_int(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(text)
    except Exception:
        return None


def load_players_from_csv(path=None):
    path = Path(path) if path else CSV_PATH
    if not path.exists():
        return []

    players = []
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_id = str(row.get("player_id") or "").strip()
                if not player_id:
                    continue

                rank = parse_optional_int(row.get("rank"))
                player_rank = parse_optional_int(row.get("player_rank"))
                if rank is None:
                    rank = player_rank

                info = enrich_rank_context(
                    rank=rank,
                    tier_key=row.get("rank_tier"),
                    explicit_weight=row.get("rank_weight"),
                )

                item = {
                    "rank": rank,
                    "player_rank": player_rank if player_rank is not None else info["player_rank"],
                    "player_id": player_id,
                    "player_name": (row.get("player_name") or "").strip(),
                    "href": (row.get("href") or f"/player/{player_id}").strip(),
                }
                item.update(info)
                players.append(item)
    except Exception as exc:
        print(f"[WARN] failed to read existing csv {path}: {exc}")
        return []

    return normalize_and_sort(players)


async def collect_all_players(context):
    page = await context.new_page()
    await setup_page(page)

    selected_players = []
    selected_ids = set()
    tier_counter = Counter()
    visited_signatures = set()
    max_rank_seen = 0

    try:
        await goto_ranking(page)

        for page_no in range(1, MAX_PAGES + 1):
            ready = await wait_for_leaderboard_ready(page, timeout_ms=30000)
            if not ready:
                print(f"[WARN] page {page_no}: leaderboard data not ready")
                await save_debug(page, f"debug_page_{page_no}_not_ready")
                break

            players = annotate_players(await extract_players_from_page(page))
            signature = tuple(p["player_id"] for p in players)

            print(f"[INFO] page {page_no}: extracted {len(players)} players")

            if not players:
                await save_debug(page, f"debug_page_{page_no}_empty_players")
                break

            if signature in visited_signatures:
                print(f"[STOP] duplicate page detected at page {page_no}")
                break

            visited_signatures.add(signature)

            page_max_rank = max((p.get("rank") or 0) for p in players)
            max_rank_seen = max(max_rank_seen, page_max_rank)
            selectable_players = get_selectable_players(players, selected_ids, tier_counter)

            if not selectable_players:
                page_window = summarize_page_window(players)
                print(
                    "[INFO] "
                    f"page {page_no}: no selectable players in ranks "
                    f"{page_window['rank_min']}-{page_window['rank_max']} "
                    f"tiers={page_window['tiers']}, skip to next page"
                )

                if tier_targets_met(tier_counter, TARGET_PLAYERS_PER_TIER) and max_rank_seen >= OTHER_TIER_MIN_RANK:
                    print(f"[INFO] reached target players for all tiers by page {page_no}")
                    break

                moved = await click_next(page)
                if not moved:
                    print("[INFO] next not available or page did not change, stop paging")
                    break

                if max_rank_seen >= MAX_RANK_TO_SCAN:
                    print(f"[INFO] stop after reaching max rank cap {max_rank_seen}")
                    break

                continue

            for player in selectable_players:
                player_id = player["player_id"]
                selected_ids.add(player_id)
                selected_players.append(player)
                tier_counter[player.get("rank_tier")] += 1

            print(
                f"[INFO] tier progress: {format_tier_progress(tier_counter)} "
                f"(selected {len(selectable_players)}/{len(players)} from page)"
            )

            if tier_targets_met(tier_counter, TARGET_PLAYERS_PER_TIER) and max_rank_seen >= OTHER_TIER_MIN_RANK:
                print(f"[INFO] reached target players for all tiers by page {page_no}")
                break

            moved = await click_next(page)
            if not moved:
                print("[INFO] next not available or page did not change, stop paging")
                break

            if max_rank_seen >= MAX_RANK_TO_SCAN:
                print(f"[INFO] stop after reaching max rank cap {max_rank_seen}")
                break

    finally:
        await page.close()

    return normalize_and_sort(selected_players)


async def save_csv(players):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            extrasaction="ignore",
            fieldnames=[
                "dataset_version",
                "rank",
                "player_rank",
                "player_id",
                "player_name",
                "href",
                "rank_tier",
                "rank_tier_label",
                "rank_weight",
            ],
        )
        writer.writeheader()
        for p in players:
            row = dict(p)
            row["dataset_version"] = SAVE_DIR.name
            writer.writerow(row)


def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat()


def parse_iso_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def load_fetch_state(path=FETCH_STATE_PATH):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to load fetch state {path}: {exc}")
        return {}
    players = data.get("players", {})
    if not isinstance(players, dict):
        return {}
    return {str(key): value for key, value in players.items() if isinstance(value, dict)}


def save_fetch_state(state, path=FETCH_STATE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": iso_now(),
        "players": state,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def html_match_count(html: str) -> int:
    if not html:
        return 0
    lower_html = html.lower()
    return lower_html.count("victory") + lower_html.count("defeat")


def html_signature(html: str) -> dict:
    normalized = html or ""
    return {
        "content_hash": hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest(),
        "match_count": html_match_count(normalized),
        "is_ready": html_has_ready_match_history(normalized),
    }


def merge_player_records(current_players, state):
    merged = {}
    for player in current_players:
        pid = str(player["player_id"])
        item = dict(player)
        item["is_current_leaderboard"] = True
        merged[pid] = item

    for pid, row in state.items():
        if pid in merged:
            continue
        rank = parse_optional_int(row.get("rank") or row.get("player_rank"))
        info = enrich_rank_context(
            rank=rank,
            tier_key=row.get("rank_tier"),
            explicit_weight=row.get("rank_weight"),
        )
        merged[pid] = {
            "rank": rank,
            "player_rank": info["player_rank"],
            "player_id": pid,
            "player_name": str(row.get("player_name") or "").strip(),
            "href": str(row.get("href") or f"/player/{pid}"),
            "is_current_leaderboard": False,
            **info,
        }
    return list(merged.values())


def revisit_interval_hours(state_row, player):
    if bool(player.get("is_current_leaderboard")):
        return 0.0
    failures = int(state_row.get("consecutive_failures", 0) or 0)
    no_change = int(state_row.get("consecutive_no_change", 0) or 0)
    recent_new_dt = parse_iso_datetime(state_row.get("last_new_match_at"))
    last_fetch_dt = parse_iso_datetime(state_row.get("last_fetch_at"))
    rank = parse_optional_int(player.get("rank") or player.get("player_rank"))

    if failures > 0:
        return min(24.0 * 14.0, DEFAULT_REVISIT_HOURS * (2 ** min(failures, 4)))
    if recent_new_dt and utc_now() - recent_new_dt <= timedelta(days=7):
        return ACTIVE_REVISIT_HOURS
    if no_change >= 3:
        return COOLDOWN_REVISIT_HOURS
    if rank is not None and rank <= 1000:
        return max(24.0, DEFAULT_REVISIT_HOURS / 2.0)
    if last_fetch_dt and utc_now() - last_fetch_dt >= timedelta(days=STALE_REVISIT_DAYS):
        return 0.0
    return DEFAULT_REVISIT_HOURS


def should_fetch_player(player, state_row):
    save_path = player_html_path(player)
    if bool(player.get("is_current_leaderboard")):
        return not (save_path.exists() and saved_player_page_is_ready(save_path))

    if not state_row:
        return True

    retry_after_dt = parse_iso_datetime(state_row.get("retry_after"))
    if retry_after_dt and utc_now() < retry_after_dt:
        return False

    last_fetch_dt = parse_iso_datetime(state_row.get("last_fetch_at"))
    if last_fetch_dt is None:
        return True

    interval_hours = revisit_interval_hours(state_row, player)
    next_fetch_dt = last_fetch_dt + timedelta(hours=interval_hours)
    return utc_now() >= next_fetch_dt


def schedule_priority(player, state_row):
    rank = parse_optional_int(player.get("rank") or player.get("player_rank")) or 10**9
    base_rank_score = max(0.0, 10000.0 - min(rank, 10000)) / 10000.0
    current_bonus = 5.0 if bool(player.get("is_current_leaderboard")) else 0.0
    failures_penalty = min(2.0, float(state_row.get("consecutive_failures", 0) or 0) * 0.25) if state_row else 0.0
    no_change_penalty = min(2.0, float(state_row.get("consecutive_no_change", 0) or 0) * 0.2) if state_row else 0.0
    recent_new_bonus = 0.0
    if state_row:
        last_new_match_dt = parse_iso_datetime(state_row.get("last_new_match_at"))
        if last_new_match_dt:
            age_hours = max(0.0, (utc_now() - last_new_match_dt).total_seconds() / 3600.0)
            recent_new_bonus = max(0.0, 3.0 - age_hours / 24.0)
    stale_bonus = 0.0
    if state_row:
        last_fetch_dt = parse_iso_datetime(state_row.get("last_fetch_at"))
        if last_fetch_dt:
            stale_bonus = min(2.0, max(0.0, (utc_now() - last_fetch_dt).total_seconds() / 86400.0) / 7.0)
    else:
        stale_bonus = 3.0
    return current_bonus + recent_new_bonus + stale_bonus + base_rank_score - failures_penalty - no_change_penalty


def sort_fetch_candidates(players):
    return sorted(
        players,
        key=lambda item: (
            -float(item.get("_priority", 0.0)),
            item.get("rank") is None,
            item.get("rank") if item.get("rank") is not None else 10**9,
            int(item["player_id"]),
        ),
    )


def tier_order():
    return [spec["key"] for spec in RANK_TIER_SPECS]


def select_balanced_current_players(candidates, max_fetches):
    if max_fetches == 0:
        return sort_fetch_candidates(candidates)

    budget = max(0, int(max_fetches))
    by_tier = {key: [] for key in tier_order()}
    for item in sort_fetch_candidates(candidates):
        by_tier.setdefault(item.get("rank_tier"), []).append(item)

    scheduled = []
    selected_ids = set()
    while len(scheduled) < budget:
        added = False
        for tier_key in tier_order():
            bucket = by_tier.get(tier_key) or []
            while bucket and bucket[0]["player_id"] in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            item = bucket.pop(0)
            scheduled.append(item)
            selected_ids.add(item["player_id"])
            added = True
            if len(scheduled) >= budget:
                break
        if not added:
            break
    return scheduled


def select_players_for_fetch(players, state, max_fetches):
    current_missing = []
    revisit_candidates = []
    for player in players:
        pid = str(player["player_id"])
        state_row = state.get(pid, {})
        if not should_fetch_player(player, state_row):
            continue
        item = dict(player)
        item["_priority"] = schedule_priority(player, state_row)
        if bool(item.get("is_current_leaderboard")):
            current_missing.append(item)
        else:
            revisit_candidates.append(item)

    scheduled = select_balanced_current_players(current_missing, max_fetches)
    selected_ids = {str(item["player_id"]) for item in scheduled}

    if max_fetches == 0:
        remaining_budget = None
    else:
        remaining_budget = max(0, int(max_fetches) - len(scheduled))

    if remaining_budget is None or remaining_budget > 0:
        revisit_sorted = [item for item in sort_fetch_candidates(revisit_candidates) if str(item["player_id"]) not in selected_ids]
        if remaining_budget is not None:
            revisit_sorted = revisit_sorted[:remaining_budget]
        scheduled.extend(revisit_sorted)
    return scheduled


def dataset_completion_by_tier(players):
    csv_counts = Counter()
    ready_counts = Counter()
    missing_counts = Counter()
    for player in players:
        if not bool(player.get("is_current_leaderboard", True)):
            continue
        tier_key = player.get("rank_tier") or enrich_rank_context(rank=player.get("rank"))["rank_tier"]
        csv_counts[tier_key] += 1
        save_path = player_html_path(player)
        if save_path.exists() and saved_player_page_is_ready(save_path):
            ready_counts[tier_key] += 1
        else:
            missing_counts[tier_key] += 1
    return csv_counts, ready_counts, missing_counts


def format_completion_counts(csv_counts, ready_counts, missing_counts):
    parts = []
    for tier_key in tier_order():
        parts.append(
            f"{tier_key}={int(ready_counts.get(tier_key, 0))}/{int(csv_counts.get(tier_key, 0))}"
            f" missing={int(missing_counts.get(tier_key, 0))}"
        )
    return ", ".join(parts)


def update_state_from_roster(state, players):
    now_text = iso_now()
    for player in players:
        pid = str(player["player_id"])
        row = dict(state.get(pid, {}))
        row.setdefault("first_seen_at", now_text)
        row["player_id"] = pid
        row["player_name"] = player.get("player_name") or row.get("player_name") or ""
        row["href"] = player.get("href") or row.get("href") or f"/player/{pid}"
        row["rank"] = player.get("rank")
        row["player_rank"] = player.get("player_rank")
        row["rank_tier"] = player.get("rank_tier")
        row["rank_tier_label"] = player.get("rank_tier_label")
        row["rank_weight"] = player.get("rank_weight")
        row["last_seen_in_leaderboard_at"] = now_text if player.get("is_current_leaderboard") else row.get("last_seen_in_leaderboard_at")
        state[pid] = row


def build_retry_after(status):
    if status in (403, 429):
        return (utc_now() + timedelta(hours=12)).isoformat()
    return (utc_now() + timedelta(hours=6)).isoformat()


def player_html_path(item):
    rank = item.get("rank")
    player_id = item["player_id"]
    rank_tier = item.get("rank_tier") or enrich_rank_context(rank=rank)["rank_tier"]
    rank_str = f"{rank:06d}" if isinstance(rank, int) else "norank"
    return SAVE_DIR / rank_tier / f"{rank_str}_{player_id}.html"


async def player_detail_worker(
    context,
    queue: asyncio.Queue,
    state,
    detail_delay_bounds,
    progress=None,
):
    page = await context.new_page()
    await setup_page(page)
    rng = random.Random(time.time())

    try:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                await asyncio.sleep(rng.uniform(*detail_delay_bounds))
                await save_player_page(page, item, state)
                if progress is not None:
                    progress.update(1)
            finally:
                queue.task_done()
    finally:
        await page.close()


async def goto_ranking(page):
    await goto_with_retry(page, RANKING_URL)
    await page.wait_for_timeout(LEADERBOARD_SETTLE_MS)

    if await is_challenge_page(page):
        cleared = await ensure_challenge_cleared(page, "first_page", auto_timeout_ms=CHALLENGE_TIMEOUT_MS)
        if not cleared:
            raise RuntimeError("排名页被 Cloudflare 验证页拦截，自动等待后仍未通过")

    unavailable_reason = leaderboard_unavailable_reason(await page_text(page))
    if "维护" in unavailable_reason:
        await save_debug(page, "debug_first_page_unavailable")
        raise RuntimeError(
            f"{unavailable_reason}。维护结束后重新执行："
            f"python build_dataset/run_all.py --download --dataset-version {SAVE_DIR.name} --force-rebuild；"
            "数据生成后再执行：python -m training.pipeline"
        )

    row_count = await wait_for_rows(page, timeout_ms=PAGE_TIMEOUT)
    if row_count == 0:
        await save_debug(page, "debug_first_page_no_rows")
        raise RuntimeError("首页未找到排行榜表格行，可能仍停留在验证页或页面结构已变化")

    ready = await wait_for_leaderboard_ready(page, timeout_ms=45000)
    if not ready:
        await save_debug(page, "debug_first_page_not_ready")
        unavailable_reason = leaderboard_unavailable_reason(await page_text(page))
        if unavailable_reason:
            raise RuntimeError(
                f"{unavailable_reason}。请稍后重新执行："
                f"python build_dataset/run_all.py --download --dataset-version {SAVE_DIR.name} --force-rebuild；"
                "数据生成后再执行：python -m training.pipeline"
            )
        raise RuntimeError("排行榜页面已打开，但等待期内没有拿到玩家列表")


async def click_next(page):
    next_btn = page.locator(NEXT_SELECTOR).first

    if await next_btn.count() == 0:
        return False

    if not await next_btn.is_visible():
        return False

    try:
        if await next_btn.is_disabled():
            return False
    except Exception:
        pass

    before_players = await extract_players_from_page(page)
    before_first_id = before_players[0]["player_id"] if before_players else None

    await next_btn.scroll_into_view_if_needed()
    await page.wait_for_timeout(LEADERBOARD_CLICK_SETTLE_MS)

    try:
        await next_btn.click(timeout=5000)
    except Exception:
        try:
            await page.evaluate("""() => {
                const btn = Array.from(document.querySelectorAll('button'))
                    .find(b => b.innerText && b.innerText.includes('Next'));
                if (btn) btn.click();
            }""")
        except Exception:
            return False

    await page.wait_for_timeout(LEADERBOARD_SETTLE_MS)

    if await is_challenge_page(page):
        cleared = await ensure_challenge_cleared(page, "next_page", auto_timeout_ms=min(45000, CHALLENGE_TIMEOUT_MS))
        if not cleared:
            return False

    ready = await wait_for_leaderboard_ready(page, timeout_ms=3000)
    if not ready:
        return False

    elapsed = 0
    step = 250
    while elapsed < 15000:
        player_link_count = await page.locator(PLAYER_LINK_SELECTOR).count()
        if player_link_count > 0:
            players = await extract_players_from_page(page)
            after_first_id = players[0]["player_id"] if players else None

            if before_first_id != after_first_id and after_first_id is not None:
                return True

        await page.wait_for_timeout(step)
        elapsed += step

    return False


async def save_player_page(page, item, state):
    rank = item.get("rank")
    player_id = item["player_id"]
    rank_tier = item.get("rank_tier") or enrich_rank_context(rank=rank)["rank_tier"]
    save_path = player_html_path(item)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pid = str(player_id)
    state_row = dict(state.get(pid, {}))
    state_row.setdefault("first_seen_at", iso_now())
    state_row["player_id"] = pid
    state_row["player_name"] = item.get("player_name") or state_row.get("player_name") or ""
    state_row["href"] = item.get("href") or state_row.get("href") or f"/player/{pid}"
    state_row["rank"] = rank
    state_row["player_rank"] = item.get("player_rank")
    state_row["rank_tier"] = rank_tier
    state_row["rank_tier_label"] = item.get("rank_tier_label")
    state_row["rank_weight"] = item.get("rank_weight")

    if save_path.exists() and saved_player_page_is_ready(save_path):
        print(f"[SKIP] rank={rank} tier={rank_tier} id={player_id}")
        state_row["last_ready_local_html_at"] = iso_now()
        state.setdefault(pid, {}).update(state_row)
        return

    try:
        response = await goto_with_retry(page, DETAIL_URL.format(player_id))
        status = response.status if response else None

        if await is_challenge_page(page):
            cleared = await ensure_challenge_cleared(
                page,
                f"detail_{player_id}",
                auto_timeout_ms=min(45000, CHALLENGE_TIMEOUT_MS),
            )
            if not cleared:
                print(f"[ERROR] rank={rank} id={player_id}: blocked by Cloudflare")
                state_row["last_http_status"] = status
                state_row["last_fetch_at"] = iso_now()
                state_row["consecutive_failures"] = int(state_row.get("consecutive_failures", 0) or 0) + 1
                state_row["retry_after"] = build_retry_after(status)
                state[pid] = state_row
                return

        if status is not None and status >= 400 and not await is_challenge_page(page):
            print(f"[{status}] rank={rank} id={player_id}")
            state_row["last_http_status"] = int(status)
            state_row["last_fetch_at"] = iso_now()
            state_row["consecutive_failures"] = int(state_row.get("consecutive_failures", 0) or 0) + 1
            state_row["retry_after"] = build_retry_after(status)
            state[pid] = state_row
            return

        html = await wait_for_match_history(page, timeout_ms=DETAIL_WAIT_MS)
        signature = html_signature(html)
        old_hash = str(state_row.get("last_content_hash") or "")
        old_match_count = int(state_row.get("last_known_match_count", 0) or 0)
        changed = signature["content_hash"] != old_hash
        has_new_matches = changed or signature["match_count"] > old_match_count

        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(html)

        print(f"[SAVED] rank={rank} tier={rank_tier} id={player_id}")
        state_row["last_http_status"] = int(status) if status is not None else 200
        state_row["last_fetch_at"] = iso_now()
        state_row["last_content_hash"] = signature["content_hash"]
        state_row["last_known_match_count"] = int(signature["match_count"])
        state_row["last_ready_html"] = bool(signature["is_ready"])
        state_row["last_ready_local_html_at"] = iso_now() if signature["is_ready"] else state_row.get("last_ready_local_html_at")
        state_row["consecutive_failures"] = 0
        state_row["retry_after"] = None
        if has_new_matches:
            state_row["last_new_match_at"] = iso_now()
            state_row["consecutive_no_change"] = 0
        else:
            state_row["consecutive_no_change"] = int(state_row.get("consecutive_no_change", 0) or 0) + 1
        state[pid] = state_row

    except PlaywrightTimeoutError:
        print(f"[ERROR] rank={rank} id={player_id}: timeout")
        state_row["last_fetch_at"] = iso_now()
        state_row["consecutive_failures"] = int(state_row.get("consecutive_failures", 0) or 0) + 1
        state_row["retry_after"] = build_retry_after(None)
        state[pid] = state_row
    except Exception as e:
        print(f"[ERROR] rank={rank} id={player_id}: {e}")
        state_row["last_fetch_at"] = iso_now()
        state_row["consecutive_failures"] = int(state_row.get("consecutive_failures", 0) or 0) + 1
        state_row["retry_after"] = build_retry_after(None)
        state[pid] = state_row


def disable_local_proxy():
    """Keep DevTools discovery off the HTTP proxy.

    A proxy in the environment turns http://[::1]:9222/json/version into a
    502, so Playwright never sees the Chrome endpoint.
    """
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    hosts = [part.strip() for part in existing.split(",") if part.strip()]
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in hosts:
            hosts.append(host)
    joined = ",".join(hosts)
    os.environ["NO_PROXY"] = joined
    os.environ["no_proxy"] = joined


def resolve_chrome_cdp_url(preferred):
    candidates = []
    for url in (preferred, "http://127.0.0.1:9333", "http://[::1]:9333"):
        if url and url not in candidates:
            candidates.append(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    errors = []
    for url in candidates:
        try:
            with opener.open(url.rstrip("/") + "/json/version", timeout=5) as response:
                info = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        browser_name = str(info.get("Browser") or "")
        if browser_name.startswith("Chrome/") and "Electron" not in browser_name:
            return url, browser_name
        errors.append(f"{url}: {browser_name or 'unknown'} 不是要连接的 Chrome")
    raise RuntimeError(
        "没有找到可用的 Chrome 调试端口。请用 9333 启动 Chrome，9222 已被 Cursor 占用。\n"
        + "\n".join(errors)
    )


async def connect_existing_chrome(playwright, cdp_url):
    try:
        browser = await playwright.chromium.connect_over_cdp(cdp_url)
    except Exception as exc:
        raise RuntimeError(
            "没有连上本机 Chrome。"
            "先关掉所有 Chrome 窗口，再执行：\n"
            "google-chrome --remote-debugging-port=9333 "
            f"--user-data-dir={Path.home() / '.config' / 'chrome-lucksack'}\n"
            "在打开的窗口里访问 https://lucksack.gg/players ，"
            "手动通过验证并确认排行榜已经显示，然后再运行下载脚本。\n"
            f"连接地址：{cdp_url}\n{exc}"
        ) from exc
    if not browser.contexts:
        raise RuntimeError("Chrome 已连接，但没有可用的浏览窗口")
    print(f"[INFO] attached to existing Chrome at {cdp_url}")
    return browser, browser.contexts[0]


async def download_players_by_rank(
    dataset_version=None,
    use_existing_csv=USE_EXISTING_VERSION_CSV,
    detail_concurrency=CONCURRENCY,
    max_detail_fetches=MAX_DETAIL_FETCHES_PER_RUN,
    detail_min_interval_seconds=DETAIL_MIN_INTERVAL_SECONDS,
    detail_max_interval_seconds=DETAIL_MAX_INTERVAL_SECONDS,
    cdp_url="http://127.0.0.1:9333",
):
    resolved_version = resolve_target_dataset_version(DATASET_ROOT, requested_version=dataset_version)
    configure_output_paths(resolved_version)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    detail_concurrency = max(1, int(detail_concurrency))
    max_detail_fetches = max(0, int(max_detail_fetches))
    detail_min_interval_seconds = max(0.0, float(detail_min_interval_seconds))
    detail_max_interval_seconds = max(detail_min_interval_seconds, float(detail_max_interval_seconds))
    fetch_state = load_fetch_state()

    print(f"[INFO] dataset version: {resolved_version}")
    print(f"[INFO] output dir: {SAVE_DIR}")
    print(f"[INFO] detail concurrency: {detail_concurrency}")
    print(f"[INFO] detail delay: {detail_min_interval_seconds:.2f}-{detail_max_interval_seconds:.2f}s")
    print(f"[INFO] max detail fetches: {max_detail_fetches}")

    players = []
    pending_players = []

    if use_existing_csv:
        players = load_players_from_csv()
        if players:
            print(f"[INFO] loaded existing csv: {CSV_PATH}")
            print(f"[INFO] total selected players from csv: {len(players)}")
            print(f"[INFO] selected tier counts: {format_tier_progress(Counter(p.get('rank_tier') for p in players))}")
    elif CSV_PATH.exists():
        print(f"[INFO] ignoring existing csv for fresh crawl: {CSV_PATH}")

    if not HEADLESS:
        disable_local_proxy()
        cdp_url, browser_name = resolve_chrome_cdp_url(cdp_url)
        print(f"[INFO] Chrome DevTools: {browser_name} at {cdp_url}")

    async with async_playwright() as p:
        browser = None
        context = None

        attached_chrome = False
        if HEADLESS:
            browser = await p.chromium.launch(headless=True)
            context = await build_context(browser)
        else:
            browser, context = await connect_existing_chrome(p, cdp_url)
            attached_chrome = True

        try:
            if not players:
                players = await collect_all_players(context)
                print(f"[INFO] total selected players: {len(players)}")

                if not players:
                    print("[WARN] no players extracted")
                    return

                await save_csv(players)
                print(f"[INFO] saved csv: {CSV_PATH}")
                print(f"[INFO] selected tier counts: {format_tier_progress(Counter(p.get('rank_tier') for p in players))}")
            merged_players = merge_player_records(players, fetch_state)
            update_state_from_roster(fetch_state, merged_players)
            csv_counts, ready_counts, missing_counts = dataset_completion_by_tier(merged_players)
            print(f"[INFO] dataset completion before fetch: {format_completion_counts(csv_counts, ready_counts, missing_counts)}")
            pending_players = select_players_for_fetch(merged_players, fetch_state, max_detail_fetches)
            current_count = sum(1 for item in pending_players if item.get("is_current_leaderboard"))
            revisit_count = len(pending_players) - current_count
            scheduled_tiers = Counter(item.get("rank_tier") for item in pending_players)
            print(f"[INFO] scheduled player pages: {len(pending_players)} current={current_count} revisit={revisit_count}")
            print(f"[INFO] scheduled tier counts: {format_tier_progress(scheduled_tiers)}")
            if pending_players:
                preview = ", ".join(
                    f"{item['player_id']}:{item.get('rank_tier')}:{item.get('_priority', 0.0):.2f}"
                    for item in pending_players[:10]
                )
                print(f"[INFO] schedule preview: {preview}")
            if not pending_players:
                print("[INFO] no scheduled player pages")
                save_fetch_state(fetch_state)
                return
            queue = asyncio.Queue()
            for item in pending_players:
                await queue.put(item)

            with tqdm(
                total=len(pending_players),
                desc="下载玩家详情",
                unit="player",
                dynamic_ncols=True,
            ) as progress:
                workers = [
                    asyncio.create_task(
                        player_detail_worker(
                            context,
                            queue,
                            fetch_state,
                            (detail_min_interval_seconds, detail_max_interval_seconds),
                            progress,
                        )
                    )
                    for _ in range(detail_concurrency)
                ]
                for _ in workers:
                    await queue.put(None)
                await queue.join()
                await asyncio.gather(*workers)
            save_fetch_state(fetch_state)
            csv_counts, ready_counts, missing_counts = dataset_completion_by_tier(merged_players)
            print(f"[INFO] dataset completion after fetch: {format_completion_counts(csv_counts, ready_counts, missing_counts)}")
        finally:
            if not attached_chrome:
                if context is not None:
                    await context.close()
                if browser is not None:
                    await browser.close()


if __name__ == "__main__":
    args = parse_args()
    HEADLESS = False if args.headful else HEADLESS
    CHALLENGE_TIMEOUT_MS = max(0, int(args.challenge_timeout_ms))
    MANUAL_CHALLENGE_TIMEOUT_MS = max(0, int(args.manual_challenge_timeout_ms))
    ENABLE_MANUAL_CHALLENGE = bool(not args.disable_manual_challenge)
    asyncio.run(
        download_players_by_rank(
            dataset_version=args.dataset_version or None,
            use_existing_csv=bool(args.use_existing_version_csv or USE_EXISTING_VERSION_CSV),
            detail_concurrency=args.detail_concurrency,
            max_detail_fetches=args.max_detail_fetches,
            detail_min_interval_seconds=args.detail_min_interval_seconds,
            detail_max_interval_seconds=args.detail_max_interval_seconds,
            cdp_url=args.cdp_url,
        )
    )
