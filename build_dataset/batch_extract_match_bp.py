import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bp_core.data_common import count_record_rows, find_existing_record_keys, upsert_record_rows
from bp_core.dataset_versions import (
    build_version_weight_map,
    describe_dataset_versions,
    extract_dataset_version_from_path,
    sort_dataset_versions,
)

from build_dataset.extract_match_bp import process_html_file

INPUT_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_RAW = str(DATA_DIR / "matches_raw.sqlite")
OUTPUT_DATASET = str(DATA_DIR / "bp_dataset.sqlite")
OUTPUT_VERSION_SUMMARY = str(PROJECT_ROOT / "json" / "dataset_version_summary.json")
PLAYER_HTML_RE = re.compile(r"^(?:(?:\d+|norank)_(\d+)|my_bp)\.html$", re.IGNORECASE)
DATASET_TABLE = "bp_dataset"
DATASET_INDEX_COLUMNS = [
    "source_html",
    "player_id",
    "rank_tier",
    "dataset_version",
]
DEFAULT_EXTRACT_WORKERS = min(8, os.cpu_count() or 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract BP matches from all versioned dataset HTML files.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-raw", default=OUTPUT_RAW)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--output-version-summary", default=OUTPUT_VERSION_SUMMARY)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--incremental", dest="incremental", action="store_true", default=True)
    parser.add_argument("--force-reextract", dest="incremental", action="store_false")
    return parser.parse_args()


def list_html_files(input_dir):
    root = Path(input_dir)
    if not root.exists():
        return []

    html_files = []
    for path in root.rglob("*.html"):
        if not path.is_file():
            continue
        if not PLAYER_HTML_RE.match(path.name):
            continue
        html_files.append(path)
    return sorted(html_files)


def handle_file(path_str, dataset_root_str, version_weights):
    raw_matches, dataset_matches = process_html_file(
        path_str,
        dataset_root=dataset_root_str,
        version_weights=version_weights,
    )
    return path_str, raw_matches, dataset_matches


def normalize_source_path(path):
    return str(path or "").replace("/", os.sep).replace("\\", os.sep)


def source_html_for_path(path):
    return os.path.relpath(str(path), start=os.getcwd())


def is_personal_bp_html(path):
    return Path(path).name.lower() == "my_bp.html"


def match_key(item):
    return (
        normalize_source_path(item.get("source_html")),
        str(item.get("page_player_id") or item.get("player_id") or ""),
        str(item.get("match_index") or ""),
    )


def match_key_text(item):
    return "\u001f".join(match_key(item))


def open_raw_match_db(path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_matches (
            key TEXT PRIMARY KEY,
            source_html TEXT NOT NULL,
            page_player_id TEXT,
            match_index TEXT,
            record BLOB NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_matches_source_html ON raw_matches(source_html)")
    return conn


def load_processed_sources_from_db(path):
    if not os.path.exists(path):
        return set(), 0
    with open_raw_match_db(path) as conn:
        rows = conn.execute("SELECT DISTINCT source_html FROM raw_matches").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
    return {normalize_source_path(row[0]) for row in rows if row[0]}, int(count)


def upsert_raw_matches(path, updates):
    if not updates:
        return 0, 0, 0
    conn = open_raw_match_db(path)
    try:
        skipped = 0
        prepared = []
        for item in updates:
            key_parts = match_key(item)
            if not key_parts[0] or not key_parts[2]:
                skipped += 1
                continue
            key = "\u001f".join(key_parts)
            prepared.append(
                (
                    key,
                    (
                        key,
                        key_parts[0],
                        key_parts[1],
                        key_parts[2],
                        sqlite3.Binary(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)),
                    ),
                )
            )

        if not prepared:
            return 0, 0, skipped

        existing_keys = find_existing_record_keys(conn, "raw_matches", [key for key, _values in prepared])
        seen_keys = set()
        appended = 0
        replaced = 0
        values = []
        for key, row_values in prepared:
            if key in existing_keys or key in seen_keys:
                replaced += 1
            else:
                appended += 1
            seen_keys.add(key)
            values.append(row_values)

        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO raw_matches
                    (key, source_html, page_player_id, match_index, record)
                VALUES (?, ?, ?, ?, ?)
                """,
                values,
            )
    finally:
        conn.close()
    return appended, replaced, skipped


def delete_html_if_no_bp(path_str, input_root, raw_matches, dataset_matches):
    if raw_matches or dataset_matches:
        return False
    if is_personal_bp_html(path_str):
        print(f"[WARN] {os.path.relpath(path_str, input_root)}: no bp data, keeping personal html")
        return False

    rel_path = os.path.relpath(path_str, input_root)
    print(f"[WARN] {rel_path}: no bp data, keeping source html")
    return False


def batch_extract_match_bp(
    input_dir=INPUT_DIR,
    output_raw=OUTPUT_RAW,
    output_dataset=OUTPUT_DATASET,
    output_version_summary=OUTPUT_VERSION_SUMMARY,
    incremental=True,
    max_workers=None,
):
    input_root = Path(input_dir)
    processed_sources, existing_raw_count = load_processed_sources_from_db(output_raw) if incremental else (set(), 0)
    discovered_html_files = list_html_files(input_root)
    dataset_db_empty = count_record_rows(output_dataset, DATASET_TABLE) == 0
    if incremental and dataset_db_empty and existing_raw_count:
        print("[INFO] dataset sqlite is empty; reparsing html to build it from existing pages")
    if incremental and not dataset_db_empty:
        html_files = [
            str(path)
            for path in discovered_html_files
            if is_personal_bp_html(path) or normalize_source_path(source_html_for_path(path)) not in processed_sources
        ]
    else:
        html_files = [str(path) for path in discovered_html_files]
    all_raw = []
    all_dataset = []
    deleted_html_count = 0

    if not html_files:
        if incremental and existing_raw_count:
            print(f"[INFO] no new html files found; keeping existing raw={existing_raw_count}")
        else:
            print("[INFO] no html files found")
        return

    discovered_versions = sort_dataset_versions(
        extract_dataset_version_from_path(path, dataset_root=input_root)
        for path in discovered_html_files
    )
    version_weights = build_version_weight_map(discovered_versions)
    version_summary = describe_dataset_versions(discovered_versions, version_weights)
    print(f"[INFO] discovered dataset versions: {version_summary}")

    worker_count = int(max_workers or DEFAULT_EXTRACT_WORKERS)
    worker_count = max(1, min(len(html_files), worker_count))

    def consume_result(path_str, raw_matches, dataset_matches):
        nonlocal deleted_html_count
        all_raw.extend(raw_matches)
        all_dataset.extend(dataset_matches)
        rel_path = os.path.relpath(path_str, input_root)
        if delete_html_if_no_bp(path_str, input_root, raw_matches, dataset_matches):
            deleted_html_count += 1
            return
        print(f"[OK] {rel_path}: {len(dataset_matches)} matches")

    retry_serial_paths = []
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_file = {
                executor.submit(handle_file, path_str, str(input_root), version_weights): path_str
                for path_str in html_files
            }

            with tqdm(
                total=len(future_to_file),
                desc="提取比赛 BP",
                unit="file",
                dynamic_ncols=True,
            ) as progress:
                for future in as_completed(future_to_file):
                    path_str = future_to_file[future]
                    try:
                        path_str, raw_matches, dataset_matches = future.result()
                        consume_result(path_str, raw_matches, dataset_matches)
                    except Exception as exc:
                        print(f"[ERR] {os.path.relpath(path_str, input_root)}: {exc}")
                        retry_serial_paths.append(path_str)
                    progress.update(1)
    except (PermissionError, OSError) as exc:
        print(f"[WARN] process pool unavailable, fallback to serial extraction: {exc}")
        retry_serial_paths = list(html_files)

    if retry_serial_paths:
        unique_retry_paths = list(dict.fromkeys(retry_serial_paths))
        print(f"[WARN] retrying {len(unique_retry_paths)} files serially after process pool errors")
        for path_str in tqdm(unique_retry_paths, desc="串行重试提取", unit="file", dynamic_ncols=True):
            try:
                path_str, raw_matches, dataset_matches = handle_file(path_str, str(input_root), version_weights)
                consume_result(path_str, raw_matches, dataset_matches)
            except Exception as inner_exc:
                print(f"[ERR] {os.path.relpath(path_str, input_root)}: {inner_exc}")

    os.makedirs("json", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if incremental:
        raw_appended, raw_replaced, raw_skipped = upsert_raw_matches(output_raw, all_raw)
        dataset_sqlite_appended, dataset_sqlite_replaced, dataset_sqlite_skipped = upsert_record_rows(
            output_dataset,
            DATASET_TABLE,
            all_dataset,
            DATASET_INDEX_COLUMNS,
            key_fn=match_key_text,
        )
    else:
        if os.path.exists(output_raw):
            os.remove(output_raw)
        if os.path.exists(output_dataset):
            os.remove(output_dataset)
        raw_appended, raw_replaced, raw_skipped = upsert_raw_matches(output_raw, all_raw)
        dataset_sqlite_appended, dataset_sqlite_replaced, dataset_sqlite_skipped = upsert_record_rows(
            output_dataset,
            DATASET_TABLE,
            all_dataset,
            DATASET_INDEX_COLUMNS,
            key_fn=match_key_text,
        )
    with open(output_version_summary, "w", encoding="utf-8") as f:
        json.dump(version_summary, f, ensure_ascii=False, indent=2)

    print(f"\nNew raw matches: {len(all_raw)}")
    print(f"New dataset matches: {len(all_dataset)}")
    _, total_raw_count = load_processed_sources_from_db(output_raw)
    print(f"Total raw matches: {total_raw_count} (+{raw_appended}, replaced {raw_replaced}, skipped {raw_skipped})")
    print(f"Dataset sqlite: +{dataset_sqlite_appended}, replaced {dataset_sqlite_replaced}, skipped {dataset_sqlite_skipped}")
    print(f"Deleted html without bp data: {deleted_html_count}")


if __name__ == "__main__":
    args = parse_args()
    batch_extract_match_bp(
        input_dir=args.input_dir,
        output_raw=args.output_raw,
        output_dataset=args.output_dataset,
        output_version_summary=args.output_version_summary,
        incremental=args.incremental,
        max_workers=args.max_workers,
    )
