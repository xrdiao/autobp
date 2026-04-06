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

async def save_page(browser, user_id: int, sem: asyncio.Semaphore):
    uid = f"{user_id:06d}"
    url = BASE_URL.format(user_id)
    save_path = SAVE_DIR / f"{uid}.html"

    if save_path.exists():
        print(f"[SKIP] {uid}")
        return

    async with sem:
        context = await browser.new_context()
        page = await context.new_page()

        try:
            response = await page.goto(url, wait_until="load", timeout=TIMEOUT)
            status = response.status if response else None

            if status != 200:
                print(f"[{status}] {uid}")
                return

            # 不等 networkidle，直接额外等待几秒让前端渲染
            await page.wait_for_timeout(3000)

            html = await page.content()

            # 可选：根据页面文本判断是否真存在
            # if "not found" in html.lower():
            #     print(f"[NOT FOUND] {uid}")
            #     return

            async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
                await f.write(html)

            print(f"[SAVED] {uid}")

        except Exception as e:
            print(f"[ERROR] {uid}: {e}")

        finally:
            await context.close()

async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        tasks = []
        for user_id in range(START_ID, END_ID + 1):
            tasks.append(asyncio.create_task(save_page(browser, user_id, sem)))

            if len(tasks) >= 100:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())