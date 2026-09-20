from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import urllib.request
import ssl

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = "https://m.swranking.com/api"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "json" / "swrt_rta_stats.json"
DEFAULT_MONSTER_LIMIT = 400
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_WORKERS = 4
NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ssl.create_default_context()),
)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def fetch_json(path, params=None):
    query = f"?{urlencode(params or {})}" if params else ""
    request = urllib.request.Request(
        f"{API_BASE}{path}{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with NO_PROXY_OPENER.open(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if int(payload.get("retCode", -1)) != 0:
        raise RuntimeError(f"SWRT request failed: {payload.get('retCode')} {path}")
    return payload.get("data")


def fetch_monster_statistics(season=None, page_size=DEFAULT_PAGE_SIZE, limit=None):
    records = []
    page_num = 1
    total = None
    target_limit = None if limit is None else max(1, int(limit))
    while total is None or len(records) < total:
        current_page_size = int(page_size)
        if target_limit is not None:
            current_page_size = min(current_page_size, target_limit - len(records))
            if current_page_size <= 0:
                break
        params = {"pageNum": page_num, "pageSize": current_page_size}
        if season is not None:
            params["season"] = season
        page = fetch_json("/monster/statistical", params)
        records.extend(page.get("list") or [])
        total = int(page.get("total") or len(records))
        if target_limit is not None and len(records) >= target_limit:
            break
        if not page.get("hasNextPage"):
            break
        page_num += 1
    return records[:target_limit] if target_limit is not None else records


def fetch_highdata(monster_id, season=None, min_interval=0.0):
    if min_interval > 0:
        time.sleep(float(min_interval))
    params = {"monsterId": monster_id}
    if season is not None:
        params["season"] = season
    return str(monster_id), fetch_json("/monster/highdata", params)


def collect_swrt_snapshot(
    season=None,
    monster_limit=DEFAULT_MONSTER_LIMIT,
    page_size=DEFAULT_PAGE_SIZE,
    max_workers=DEFAULT_MAX_WORKERS,
    request_interval=0.0,
):
    statistics = fetch_monster_statistics(
        season=season,
        page_size=page_size,
        limit=max(1, int(monster_limit)),
    )
    ranked = sorted(
        statistics,
        key=lambda item: int(item.get("pickTotal") or 0),
        reverse=True,
    )
    highdata_targets = [item for item in ranked[: max(0, int(monster_limit))] if item.get("monsterId")]
    highdata_by_id = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [
            executor.submit(
                fetch_highdata,
                item["monsterId"],
                season,
                request_interval,
            )
            for item in highdata_targets
        ]
        with tqdm(
            total=len(futures),
            desc="SWRT highdata",
            unit="monster",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                monster_id, data = future.result()
                highdata_by_id[monster_id] = data
                progress.update(1)

    return {
        "schema_version": 1,
        "source": "swranking.com",
        "source_urls": {
            "statistics": f"{API_BASE}/monster/statistical",
            "highdata": f"{API_BASE}/monster/highdata",
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "monster_count": len(statistics),
        "highdata_limit": len(highdata_targets),
        "statistics": statistics,
        "statistics_by_id": {
            str(item.get("monsterId")): item
            for item in statistics
            if item.get("monsterId")
        },
        "highdata_by_id": highdata_by_id,
    }


def sync_swrt_data(
    output_path=DEFAULT_OUTPUT_PATH,
    season=None,
    monster_limit=DEFAULT_MONSTER_LIMIT,
    page_size=DEFAULT_PAGE_SIZE,
    max_workers=DEFAULT_MAX_WORKERS,
    request_interval=0.0,
):
    snapshot = collect_swrt_snapshot(
        season=season,
        monster_limit=monster_limit,
        page_size=page_size,
        max_workers=max_workers,
        request_interval=request_interval,
    )
    save_json(snapshot, output_path)
    print(
        f"[OK] saved SWRT stats: {output_path} "
        f"({snapshot['monster_count']} monsters, {snapshot['highdata_limit']} highdata)"
    )
    return snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Sync public SWRT RTA monster and matchup statistics.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--season", default=None, help="Optional SWRT season id. Omit to use SWRT current data.")
    parser.add_argument("--monster-limit", type=int, default=DEFAULT_MONSTER_LIMIT)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--request-interval", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    sync_swrt_data(
        output_path=args.output_path,
        season=args.season,
        monster_limit=args.monster_limit,
        page_size=args.page_size,
        max_workers=args.max_workers,
        request_interval=args.request_interval,
    )


if __name__ == "__main__":
    main()
