import asyncio
from pathlib import Path

import aiofiles
from playwright.async_api import async_playwright

SAVE_DIR = Path("./dataset")
BASE_URL = "https://lucksack.gg/player/{}"

# 先小范围测试
START_ID = 174831
END_ID = 174940

CONCURRENCY = 3
TIMEOUT = 30000
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}


async def setup_page(page):
    async def handle_route(route):
        if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
            await route.abort()
            return
        await route.continue_()

    await page.route("**/*", handle_route)


async def save_page(page, user_id: int):
    uid = f"{user_id:06d}"
    url = BASE_URL.format(user_id)
    save_path = SAVE_DIR / f"{uid}.html"

    if save_path.exists():
        print(f"[SKIP] {uid}")
        return

    try:
        response = await page.goto(url, wait_until="load", timeout=TIMEOUT)
        status = response.status if response else None

        if status != 200:
            print(f"[{status}] {uid}")
            return

        # 不等 networkidle，额外等待几秒让前端渲染
        await page.wait_for_timeout(3000)
        html = await page.content()

        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(html)

        print(f"[SAVED] {uid}")

    except Exception as e:
        print(f"[ERROR] {uid}: {e}")


async def page_worker(browser, queue: asyncio.Queue):
    context = await browser.new_context()
    page = await context.new_page()
    await setup_page(page)

    try:
        while True:
            user_id = await queue.get()
            try:
                if user_id is None:
                    return
                await save_page(page, user_id)
            finally:
                queue.task_done()
    finally:
        await context.close()


async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    pending_ids = [
        user_id
        for user_id in range(START_ID, END_ID + 1)
        if not (SAVE_DIR / f"{user_id:06d}.html").exists()
    ]

    if not pending_ids:
        print("[INFO] no pending pages")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        queue = asyncio.Queue()

        for user_id in pending_ids:
            await queue.put(user_id)

        workers = [asyncio.create_task(page_worker(browser, queue)) for _ in range(CONCURRENCY)]
        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
