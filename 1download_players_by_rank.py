import asyncio
import csv
import re
from pathlib import Path

import aiofiles
from playwright.async_api import async_playwright

RANKING_URL = "https://lucksack.gg/players"
DETAIL_URL = "https://lucksack.gg/player/{}"

SAVE_DIR = Path("./dataset")
CSV_PATH = SAVE_DIR / "ranking_list.csv"

PAGE_TIMEOUT = 30000
DETAIL_WAIT_MS = 3000
CONCURRENCY = 3
MAX_PAGES = 6

ROW_SELECTOR = 'tbody[data-slot="table-body"] tr[data-slot="table-row"]'
NEXT_SELECTOR = 'button:has-text("Next")'

async def save_debug(page, name):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        html = await page.content()
        (SAVE_DIR / f"{name}.html").write_text(html, encoding="utf-8")
    except:
        pass
    try:
        await page.screenshot(path=str(SAVE_DIR / f"{name}.png"), full_page=True)
    except:
        pass

async def wait_for_rows(page, timeout_ms=30000):
    elapsed = 0
    step = 1000

    while elapsed < timeout_ms:
        count = await page.locator(ROW_SELECTOR).count()
        if count > 0:
            return count

        try:
            await page.mouse.wheel(0, 1500)
        except:
            pass

        await page.wait_for_timeout(step)
        elapsed += step

    return 0

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

async def goto_ranking(page):
    await page.goto(RANKING_URL, wait_until="load", timeout=PAGE_TIMEOUT)
    await page.wait_for_timeout(2000)

    row_count = await wait_for_rows(page, timeout_ms=PAGE_TIMEOUT)
    if row_count == 0:
        await save_debug(page, "debug_first_page_no_rows")
        raise RuntimeError("首页未找到表格行")

async def click_next(page):
    next_btn = page.locator(NEXT_SELECTOR).first

    if await next_btn.count() == 0:
        return False

    if not await next_btn.is_visible():
        return False

    try:
        disabled = await next_btn.is_disabled()
        if disabled:
            return False
    except:
        pass

    before_players = await extract_players_from_page(page)
    before_first_id = before_players[0]["player_id"] if before_players else None

    await next_btn.scroll_into_view_if_needed()
    await page.wait_for_timeout(500)

    try:
        await next_btn.click(timeout=5000)
    except:
        try:
            await page.evaluate("""() => {
                const btn = Array.from(document.querySelectorAll('button'))
                    .find(b => b.innerText && b.innerText.includes('Next'));
                if (btn) btn.click();
            }""")
        except:
            return False

    await page.wait_for_timeout(2000)

    # 等待内容变化
    elapsed = 0
    step = 1000
    while elapsed < 15000:
        row_count = await page.locator(ROW_SELECTOR).count()
        if row_count > 0:
            players = await extract_players_from_page(page)
            after_first_id = players[0]["player_id"] if players else None

            if before_first_id != after_first_id and after_first_id is not None:
                return True

        await page.wait_for_timeout(step)
        elapsed += step

    return False

async def collect_all_players(browser):
    context = await browser.new_context()
    page = await context.new_page()

    all_players = []
    visited_signatures = set()

    try:
        await goto_ranking(page)

        for page_no in range(1, MAX_PAGES + 1):
            row_count = await wait_for_rows(page, timeout_ms=10000)
            if row_count == 0:
                print(f"[WARN] page {page_no}: no rows found")
                await save_debug(page, f"debug_page_{page_no}_no_rows")
                break

            players = await extract_players_from_page(page)
            signature = tuple(p["player_id"] for p in players)

            print(f"[INFO] page {page_no}: extracted {len(players)} players")

            if not players:
                await save_debug(page, f"debug_page_{page_no}_empty_players")
                break

            if signature in visited_signatures:
                print(f"[STOP] duplicate page detected at page {page_no}")
                break

            visited_signatures.add(signature)
            all_players.extend(players)

            moved = await click_next(page)
            if not moved:
                print("[INFO] next not available or page did not change, stop paging")
                break

    finally:
        await context.close()

    return normalize_and_sort(all_players)

async def save_csv(players):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "player_id", "player_name", "href"])
        writer.writeheader()
        for p in players:
            writer.writerow(p)

async def save_player_page(browser, sem, rank, player_id):
    async with sem:
        rank_str = f"{rank:06d}" if isinstance(rank, int) else "norank"
        save_path = SAVE_DIR / f"{rank_str}_{player_id}.html"

        if save_path.exists():
            print(f"[SKIP] rank={rank} id={player_id}")
            return

        context = await browser.new_context()
        page = await context.new_page()

        try:
            response = await page.goto(DETAIL_URL.format(player_id), wait_until="load", timeout=PAGE_TIMEOUT)
            status = response.status if response else None

            if status != 200:
                print(f"[{status}] rank={rank} id={player_id}")
                return

            await page.wait_for_timeout(DETAIL_WAIT_MS)
            html = await page.content()

            async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
                await f.write(html)

            print(f"[SAVED] rank={rank} id={player_id}")

        except Exception as e:
            print(f"[ERROR] rank={rank} id={player_id}: {e}")

        finally:
            await context.close()

async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        players = await collect_all_players(browser)
        print(f"[INFO] total unique players: {len(players)}")

        if not players:
            print("[WARN] no players extracted")
            await browser.close()
            return

        await save_csv(players)
        print(f"[INFO] saved csv: {CSV_PATH}")

        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            asyncio.create_task(save_player_page(browser, sem, item["rank"], item["player_id"]))
            for item in players
        ]
        await asyncio.gather(*tasks)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())