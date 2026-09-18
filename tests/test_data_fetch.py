import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from build_dataset.download_players_by_rank import wait_for_match_history
from build_dataset.sync_lucksack_data import fetch_rank_group


class FakeDetailPage:
    def __init__(self, html_rows):
        self.html_rows = list(html_rows)
        self.wait_calls = []

    async def content(self):
        return self.html_rows[min(len(self.wait_calls), len(self.html_rows) - 1)]

    async def wait_for_timeout(self, milliseconds):
        self.wait_calls.append(milliseconds)


class DataFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_detail_wait_returns_as_soon_as_history_is_ready(self):
        incomplete = "<html><body>Loading...</body></html>"
        ready = "<html>Match History " + ("/monster/x " * 10) + " victory</html>"
        page = FakeDetailPage([incomplete, incomplete, ready])

        result = await wait_for_match_history(page, timeout_ms=1000)

        self.assertEqual(result, ready)
        self.assertEqual(page.wait_calls, [250, 250])

    async def test_lucksack_rank_group_fetch_keeps_all_sources(self):
        with patch(
            "build_dataset.sync_lucksack_data.fetch_statistics",
            new=AsyncMock(side_effect=[["solo"], ["duo"], ["trio"]]),
        ) as stats_fetch, patch(
            "build_dataset.sync_lucksack_data.fetch_json",
            new=AsyncMock(side_effect=[{"summary": True}, {"draft": True}]),
        ) as json_fetch:
            result = await fetch_rank_group(
                object(),
                {"group_id": 102, "group_name": "Guardian"},
                38,
                10,
                2000,
                asyncio.Semaphore(1),
            )

        self.assertEqual(result, ("102", ["solo"], ["duo"], {"summary": True}, {"draft": True}, ["trio"]))
        self.assertEqual(stats_fetch.await_count, 3)
        self.assertEqual(json_fetch.await_count, 2)


if __name__ == "__main__":
    unittest.main()
