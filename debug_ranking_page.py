import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

SAVE_DIR = Path("./dataset")
RANKING_URL = "https://lucksack.gg/players"

async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # 先开可视模式排查
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(RANKING_URL, wait_until="load", timeout=30000)
        await page.wait_for_timeout(5000)

        # 保存首页源码和截图
        html = await page.content()
        (SAVE_DIR / "debug_home.html").write_text(html, encoding="utf-8")
        await page.screenshot(path=str(SAVE_DIR / "debug_home.png"), full_page=True)

        print("当前 URL:", page.url)

        # 检查玩家链接数量
        count_player_links = await page.locator('a[href^="/player/"]').count()
        print("主页面 /player/ 链接数量:", count_player_links)

        # 检查表格行数量
        count_rows = await page.locator('tbody tr').count()
        print("主页面 tbody tr 数量:", count_rows)

        # 检查 data-slot 结构
        count_slot_rows = await page.locator('tbody[data-slot="table-body"] tr[data-slot="table-row"]').count()
        print('主页面 data-slot rows 数量:', count_slot_rows)

        # 检查按钮
        buttons = await page.locator("button").all_text_contents()
        print("页面按钮文本：")
        for i, b in enumerate(buttons[:30], 1):
            print(i, repr(b.strip()))

        # 检查 iframe
        frames = page.frames
        print("iframe/frame 数量:", len(frames))

        for idx, frame in enumerate(frames):
            try:
                url = frame.url
                link_count = await frame.locator('a[href^="/player/"]').count()
                row_count = await frame.locator('tbody tr').count()
                slot_row_count = await frame.locator('tbody[data-slot="table-body"] tr[data-slot="table-row"]').count()
                print(f"[Frame {idx}] url={url}")
                print(f"  /player/ links = {link_count}")
                print(f"  tbody tr = {row_count}")
                print(f"  slot rows = {slot_row_count}")
            except Exception as e:
                print(f"[Frame {idx}] error: {e}")

        await page.pause()  # 让你手动看页面，调试用
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())