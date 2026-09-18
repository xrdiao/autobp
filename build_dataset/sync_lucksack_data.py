from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from playwright.async_api import async_playwright
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT_URL = "https://lucksack.gg/"
API_BASE = "https://api.lucksack.gg"
CATALOG_URL = "https://static.lucksack.gg/data/monsters_catalog.json"
DEFAULT_STATS_PATH = PROJECT_ROOT / "json" / "lucksack_stats.json"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "json" / "lucksack_monster_catalog.json"
DEFAULT_IMAGE_MAP_PATH = PROJECT_ROOT / "json" / "monster_images.json"
DEFAULT_COMBO_LIMIT = 2000
STATISTICS_LIMIT = 10000
DEFAULT_GROUP_CONCURRENCY = 2


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def update_image_map_from_catalog(catalog, image_map_path=DEFAULT_IMAGE_MAP_PATH):
    image_map_path = Path(image_map_path)
    existing = {}
    if image_map_path.exists():
        try:
            with open(image_map_path, "r", encoding="utf-8") as file:
                value = json.load(file)
            if isinstance(value, dict):
                existing = value
        except (OSError, json.JSONDecodeError):
            existing = {}

    for item in catalog or []:
        name = str(
            item.get("slabel")
            or item.get("label")
            or item.get("slug")
            or ""
        ).strip()
        image = str(item.get("image") or "").strip()
        if not name or not image:
            continue
        if not image.startswith(("http://", "https://")):
            image = f"https://static.lucksack.gg/images/monsters/{image.lstrip('/')}"
        existing.setdefault(name, image)
    save_json(existing, image_map_path)
    return existing


async def fetch_json(page, url):
    payload = await page.evaluate(
        """
        async (url) => {
            const response = await fetch(url);
            return {
                status: response.status,
                text: await response.text(),
            };
        }
        """,
        url,
    )
    if int(payload["status"]) != 200:
        raise RuntimeError(f"request failed: {payload['status']} {url}")
    return json.loads(payload["text"])


def choose_active_season(seasons):
    active = [item for item in seasons if item.get("is_active") and item.get("season_number")]
    if not active:
        active = [item for item in seasons if item.get("season_number")]
    return max(active, key=lambda item: int(item["season_number"]))


def choose_active_patch(patches):
    active = [item for item in patches if not item.get("ends_at")]
    if active:
        return max(active, key=lambda item: int(item.get("patch_order") or 0))
    return max(patches, key=lambda item: int(item.get("patch_order") or 0))


def build_statistics_url(kind, season_number, rank_group, patch_id, limit, offset):
    query = {
        "season": season_number,
        "rank": rank_group,
        "limit": min(STATISTICS_LIMIT, int(limit)),
        "offset": int(offset),
        "order_by": "Pick",
        "order_direction": "Desc",
    }
    if patch_id is not None:
        query["patch"] = patch_id
    return f"{API_BASE}/statistics/{kind}?{urlencode(query)}"


def build_meta_dashboard_url(path, season_number, rank_group, patch_id):
    query = {
        "season": season_number,
        "rank": rank_group,
    }
    if patch_id is not None:
        query["patch"] = patch_id
    return f"{API_BASE}/meta-dashboard/{path}?{urlencode(query)}"


async def fetch_statistics(page, kind, season_number, rank_group, patch_id, target_limit):
    target_limit = max(1, min(STATISTICS_LIMIT, int(target_limit)))
    first = await fetch_json(
        page,
        build_statistics_url(
            kind,
            season_number,
            rank_group,
            patch_id,
            target_limit,
            0,
        ),
    )
    records = list(first.get("records") or [])
    total = int(first.get("count") or len(records))
    while len(records) < min(total, target_limit):
        offset = len(records)
        page_data = await fetch_json(
            page,
            build_statistics_url(
                kind,
                season_number,
                rank_group,
                patch_id,
                target_limit - offset,
                offset,
            ),
        )
        new_records = list(page_data.get("records") or [])
        if not new_records:
            break
        records.extend(new_records)
    return records[:target_limit]


async def fetch_rank_group(page, rank_group, season_number, patch_id, combo_limit, semaphore):
    group_id = str(rank_group["group_id"])
    async with semaphore:
        print(f"[INFO] syncing rank group {group_id}: {rank_group.get('group_name')}")
        solo = await fetch_statistics(
            page,
            "solo",
            season_number,
            group_id,
            patch_id,
            STATISTICS_LIMIT,
        )
        duo = await fetch_statistics(
            page,
            "duo",
            season_number,
            group_id,
            patch_id,
            combo_limit,
        )
        summary = await fetch_json(
            page,
            build_meta_dashboard_url(
                "summary",
                season_number,
                group_id,
                patch_id,
            ),
        )
        draft_position = await fetch_json(
            page,
            build_meta_dashboard_url(
                "draft-position",
                season_number,
                group_id,
                patch_id,
            ),
        )
        trio = await fetch_statistics(
            page,
            "trio",
            season_number,
            group_id,
            patch_id,
            combo_limit,
        )
    return group_id, solo, duo, summary, draft_position, trio


async def collect_snapshot(
    combo_limit=DEFAULT_COMBO_LIMIT,
    group_concurrency=DEFAULT_GROUP_CONCURRENCY,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(ROOT_URL, wait_until="domcontentloaded", timeout=60000)

        seasons = await fetch_json(page, f"{API_BASE}/seasons")
        season = choose_active_season(seasons)
        season_number = int(season["season_number"])
        rank_groups = await fetch_json(page, f"{API_BASE}/seasons/{season_number}/rank-groups")
        patches = await fetch_json(page, f"{API_BASE}/seasons/{season_number}/patches")
        patch = choose_active_patch(patches)
        patch_id = patch.get("patch_id")
        catalog = await fetch_json(page, CATALOG_URL)
        rank_distribution = await fetch_json(
            page,
            f"{API_BASE}/statistics/ranks?{urlencode({'season': season_number})}",
        )

        statistics = {"solo": {}, "duo": {}, "trio": {}}
        meta_summary = {}
        meta_draft_position = {}
        semaphore = asyncio.Semaphore(max(1, int(group_concurrency)))
        tasks = [
            fetch_rank_group(
                page,
                rank_group,
                season_number,
                patch_id,
                combo_limit,
                semaphore,
            )
            for rank_group in rank_groups
        ]
        results = []
        with tqdm(
            total=len(tasks),
            desc="Lucksack rank groups",
            unit="group",
            dynamic_ncols=True,
        ) as progress:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                progress.update(1)
                progress.set_postfix_str(f"group={result[0]}")
        for group_id, solo, duo, summary, draft_position, trio in results:
            statistics["solo"][group_id] = solo
            statistics["duo"][group_id] = duo
            meta_summary[group_id] = summary
            meta_draft_position[group_id] = draft_position
            statistics["trio"][group_id] = trio

        await browser.close()

    return {
        "schema_version": 2,
        "source": "lucksack.gg",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "patch": patch,
        "default_rank_group": "102",
        "rank_groups": rank_groups,
        "patches": patches,
        "meta_dashboard": {
            "rank_distribution": rank_distribution.get("rank_distribution", rank_distribution),
            "summary": meta_summary,
            "draft_position": meta_draft_position,
        },
        "catalog_count": len(catalog),
        "catalog": catalog,
        "statistics": statistics,
    }


async def sync_lucksack_data(
    stats_path=DEFAULT_STATS_PATH,
    catalog_path=DEFAULT_CATALOG_PATH,
    combo_limit=DEFAULT_COMBO_LIMIT,
    group_concurrency=DEFAULT_GROUP_CONCURRENCY,
):
    snapshot = await collect_snapshot(
        combo_limit=combo_limit,
        group_concurrency=group_concurrency,
    )
    save_json(snapshot, stats_path)
    save_json(snapshot["catalog"], catalog_path)
    image_map = update_image_map_from_catalog(snapshot["catalog"])
    print(f"[OK] saved stats: {stats_path}")
    print(f"[OK] saved catalog: {catalog_path} ({len(snapshot['catalog'])} entries)")
    print(f"[OK] completed image map: {DEFAULT_IMAGE_MAP_PATH} ({len(image_map)} entries)")
    print(
        "[OK] season="
        f"{snapshot['season'].get('season_number')} "
        f"patch={snapshot['patch'].get('patch_id')} "
        f"rank_groups={len(snapshot['rank_groups'])}"
    )
    return snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Sync Lucksack public RTA statistics and monster catalog.")
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS_PATH))
    parser.add_argument("--catalog-path", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--combo-limit", type=int, default=DEFAULT_COMBO_LIMIT)
    parser.add_argument("--group-concurrency", type=int, default=DEFAULT_GROUP_CONCURRENCY)
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(
        sync_lucksack_data(
            stats_path=args.stats_path,
            catalog_path=args.catalog_path,
            combo_limit=args.combo_limit,
            group_concurrency=args.group_concurrency,
        )
    )


if __name__ == "__main__":
    main()
