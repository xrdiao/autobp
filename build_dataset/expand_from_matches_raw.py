import json
import os
import argparse
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bp_core.data_common import count_record_rows, load_json, load_pt, save_json, save_pt, upsert_record_rows

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DB_PATH = DATA_DIR / "matches_raw.sqlite"
DEFAULT_OUTPUT_DB_PATH = DATA_DIR / "bp_samples.sqlite"
DEFAULT_ALL_SAMPLES_PATH = ""
DEFAULT_ME_ONLY_SAMPLES_PATH = ""
DEFAULT_MANIFEST_PATH = DATA_DIR / "bp_training_samples_manifest.json"
DEFAULT_FETCH_BATCH_SIZE = 1000
SAMPLE_TABLE = "bp_samples"
SAMPLE_INDEX_COLUMNS = [
    "sample_id",
    "match_id",
    "player_id",
    "current_side",
    "rank_tier",
    "dataset_version",
    "source_html",
]


def iter_raw_matches_from_sqlite(path: str, batch_size: int = DEFAULT_FETCH_BATCH_SIZE):
    if not os.path.exists(path):
        return
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute("SELECT key, record FROM raw_matches ORDER BY key")
        while True:
            rows = cursor.fetchmany(max(1, int(batch_size)))
            if not rows:
                break
            for key, record in rows:
                try:
                    match = pickle.loads(record)
                except Exception as exc:
                    print(f"[WARN] failed to decode raw match {key}: {exc}", flush=True)
                    continue
                yield key, match
    finally:
        conn.close()


def load_jsonl(path: str):
    if not os.path.exists(path):
        return []

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping broken JSONL line {line_no} in {path}: {exc}", flush=True)
    return data


def load_samples(json_path: str, jsonl_path: str):
    try:
        return load_json(json_path, default=[])
    except json.JSONDecodeError as exc:
        print(f"[WARN] {json_path} is not valid JSON: {exc}", flush=True)
        print(f"[WARN] Recovering existing samples from {jsonl_path}", flush=True)
        return load_jsonl(jsonl_path)


def save_with_log(label: str, save_func, data, path: str):
    if not path:
        return
    started = time.perf_counter()
    print(f"Saving {label}: {path} ({len(data)} rows)...", flush=True)
    save_func(data, path)
    elapsed = time.perf_counter() - started
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved {label}: {size_mb:.1f} MB in {elapsed:.1f}s", flush=True)

def result_to_win_flag(result) -> bool | None:
    text = str(result or "").strip().lower()
    if text in {"win", "victory", "winner"}:
        return True
    if text in {"lose", "loss", "defeat"}:
        return False
    return None

def get_side_value(picks: list, field_name: str, value_name: str):
    for x in picks:
        if x.get(field_name):
            return x.get(value_name)
    return None

def build_combined_sequence(match: dict):
    combined = []
    for side_key, side_name in (("my_picks", "me"), ("enemy_picks", "enemy")):
        for item in match.get(side_key, []):
            order = item.get("order")
            if order is None:
                continue
            combined.append({
                "side": side_name,
                "order": order,
                "name": item.get("name"),
                "slug": item.get("slug"),
            })
    combined.sort(key=lambda x: x["order"])
    return combined

def validate_match_sequence(combined_sequence: list) -> bool:
    if len(combined_sequence) != 10:
        return False
    return all(item["order"] == i for i, item in enumerate(combined_sequence, start=1))

def expand_one_match(match: dict):
    combined_sequence = build_combined_sequence(match)
    if not validate_match_sequence(combined_sequence):
        return None

    first_pick_side = combined_sequence[0]["side"]

    my_picks_raw = match.get("my_picks", [])
    enemy_picks_raw = match.get("enemy_picks", [])

    my_ban = get_side_value(my_picks_raw, "is_banned", "name")
    enemy_ban = get_side_value(enemy_picks_raw, "is_banned", "name")
    my_leader = get_side_value(my_picks_raw, "is_leader", "name")
    enemy_leader = get_side_value(enemy_picks_raw, "is_leader", "name")
    my_ban_slug = get_side_value(my_picks_raw, "is_banned", "slug")
    enemy_ban_slug = get_side_value(enemy_picks_raw, "is_banned", "slug")
    my_leader_slug = get_side_value(my_picks_raw, "is_leader", "slug")
    enemy_leader_slug = get_side_value(enemy_picks_raw, "is_leader", "slug")

    player_id = match.get("page_player_id")
    player_rank = match.get("player_rank")
    match_index = match.get("match_index")
    match_id = f"{player_id}_{match_index}"
    rank_tier = match.get("rank_tier")
    rank_tier_label = match.get("rank_tier_label")
    rank_weight = match.get("rank_weight")
    dataset_version = match.get("dataset_version")
    version_weight = match.get("version_weight")
    sample_weight = match.get("sample_weight")
    source_html = match.get("source_html")

    result = match.get("result")
    time_ago = match.get("time_ago")
    self_info = match.get("self") or {}
    opponent_info = match.get("opponent") or {}

    samples = []
    me_only_samples = []
    my_sequence = [item for item in combined_sequence if item.get("side") == "me"]
    enemy_sequence = [item for item in combined_sequence if item.get("side") == "enemy"]

    def build_remaining_values(sequence, key):
        remaining = [[] for _ in range(len(sequence) + 1)]
        for index in range(len(sequence) - 1, -1, -1):
            values = remaining[index + 1].copy()
            value = sequence[index].get(key)
            if value:
                values.insert(0, value)
            remaining[index] = values
        return remaining

    my_remaining_picks = build_remaining_values(my_sequence, "name")
    enemy_remaining_picks = build_remaining_values(enemy_sequence, "name")
    my_remaining_slugs = build_remaining_values(my_sequence, "slug")
    enemy_remaining_slugs = build_remaining_values(enemy_sequence, "slug")
    final_my_picks = my_remaining_picks[0]
    final_enemy_picks = enemy_remaining_picks[0]
    final_my_pick_slugs = my_remaining_slugs[0]
    final_enemy_pick_slugs = enemy_remaining_slugs[0]
    my_current = []
    enemy_current = []
    my_current_slugs = []
    enemy_current_slugs = []
    my_seen = 0
    enemy_seen = 0

    for step in combined_sequence:
        future_info = {
            "remaining_my_picks": my_remaining_picks[my_seen],
            "remaining_enemy_picks": enemy_remaining_picks[enemy_seen],
            "remaining_my_pick_slugs": my_remaining_slugs[my_seen],
            "remaining_enemy_pick_slugs": enemy_remaining_slugs[enemy_seen],
        }
        sample = {
            "sample_id": f"{match_id}_{step['order']}",
            "match_id": match_id,
            "player_id": player_id,
            "player_rank": player_rank,
            "rank_tier": rank_tier,
            "rank_tier_label": rank_tier_label,
            "rank_weight": rank_weight,
            "dataset_version": dataset_version,
            "version_weight": version_weight,
            "sample_weight": sample_weight,
            "match_index": match_index,
            "result": result,
            "won": result_to_win_flag(result),
            "time_ago": time_ago,
            "source_html": source_html,
            "self_name": self_info.get("name"),
            "self_rating": self_info.get("rating"),
            "opponent_name": opponent_info.get("name"),
            "opponent_rating": opponent_info.get("rating"),
            "turn": step["order"],
            "first_pick_side": first_pick_side,
            "current_side": step["side"],
            "my_picks": my_current.copy(),
            "my_picks_slugs": my_current_slugs.copy(),
            "enemy_picks": enemy_current.copy(),
            "enemy_picks_slugs": enemy_current_slugs.copy(),
            "my_ban": my_ban,
            "my_ban_slug": my_ban_slug,
            "enemy_ban": enemy_ban,
            "enemy_ban_slug": enemy_ban_slug,
            "my_leader": my_leader,
            "my_leader_slug": my_leader_slug,
            "enemy_leader": enemy_leader,
            "enemy_leader_slug": enemy_leader_slug,
            "next_pick": step["name"],
            "next_pick_slug": step.get("slug"),
            "remaining_my_picks": future_info["remaining_my_picks"],
            "remaining_enemy_picks": future_info["remaining_enemy_picks"],
            "remaining_my_pick_slugs": future_info["remaining_my_pick_slugs"],
            "remaining_enemy_pick_slugs": future_info["remaining_enemy_pick_slugs"],
            "final_my_picks": final_my_picks,
            "final_enemy_picks": final_enemy_picks,
            "final_my_pick_slugs": final_my_pick_slugs,
            "final_enemy_pick_slugs": final_enemy_pick_slugs,
        }
        samples.append(sample)

        if step["side"] == "me":
            me_only_samples.append(sample)
            my_current.append(step["name"])
            if step.get("slug"):
                my_current_slugs.append(step["slug"])
            my_seen += 1
        else:
            enemy_current.append(step["name"])
            if step.get("slug"):
                enemy_current_slugs.append(step["slug"])
            enemy_seen += 1

    return samples, me_only_samples

def merge_samples(existing, updates, key_name):
    merged = {str(item.get(key_name)): item for item in existing if item.get(key_name)}
    appended = 0
    replaced = 0
    for item in updates:
        key = str(item.get(key_name) or "")
        if not key:
            continue
        if key in merged:
            replaced += 1
        else:
            appended += 1
        merged[key] = item
    return list(merged.values()), appended, replaced


def raw_match_id(match):
    return f"{match.get('page_player_id')}_{match.get('match_index')}"


def load_manifest(path: str):
    manifest = load_json(path, default={})
    if not isinstance(manifest, dict):
        return {"processed_match_ids": [], "stats": {}}
    processed = manifest.get("processed_match_ids") or []
    stats = manifest.get("stats") or {}
    return {
        "processed_match_ids": [str(item) for item in processed if item],
        "stats": stats if isinstance(stats, dict) else {},
    }


def save_manifest(path: str, processed_match_ids, stats=None):
    save_json(
        {
            "processed_match_ids": sorted(str(item) for item in processed_match_ids if item),
            "stats": stats or {},
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
        indent=2,
    )


def expand_from_matches_raw(
    incremental=True,
    input_path=str(DEFAULT_RAW_DB_PATH),
    output_db_path=str(DEFAULT_OUTPUT_DB_PATH),
    output_all_path=str(DEFAULT_ALL_SAMPLES_PATH),
    output_me_only_path=str(DEFAULT_ME_ONLY_SAMPLES_PATH),
    manifest_path=str(DEFAULT_MANIFEST_PATH),
    fetch_batch_size=DEFAULT_FETCH_BATCH_SIZE,
    max_workers=None,
    export_pt=False,
):
    manifest = load_manifest(manifest_path) if incremental else {"processed_match_ids": []}
    processed_match_ids = set(manifest.get("processed_match_ids") or [])
    if incremental and processed_match_ids and count_record_rows(output_db_path, SAMPLE_TABLE) == 0:
        print("[INFO] sample sqlite is empty; rebuilding from raw matches instead of trusting the old manifest")
        processed_match_ids = set()
    pending_matches = []
    total_raw_matches = 0
    for _key, match in iter_raw_matches_from_sqlite(input_path, batch_size=fetch_batch_size):
        total_raw_matches += 1
        match_id = raw_match_id(match)
        if incremental and match_id in processed_match_ids:
            continue
        pending_matches.append(match)
    if not processed_match_ids and incremental and pending_matches and output_all_path and os.path.exists(output_all_path):
        existing_all_for_manifest = load_pt(output_all_path, default=[])
        processed_match_ids = {
            str(item.get("match_id"))
            for item in existing_all_for_manifest
            if item.get("match_id")
        }
        pending_matches = [
            match for match in pending_matches
            if raw_match_id(match) not in processed_match_ids
        ]
        existing_all_for_manifest = None

    if incremental and not pending_matches:
        stats = dict(manifest.get("stats") or {})
        stats.update(
            {
                "raw_matches": int(total_raw_matches),
                "pending_matches": 0,
                "valid_newly_expanded_matches": 0,
                "skipped_newly_expanded_matches": 0,
            }
        )
        save_manifest(manifest_path, processed_match_ids, stats=stats)
        existing_all_count = count_record_rows(output_db_path, SAMPLE_TABLE)
        existing_me_only_count = count_record_rows(output_db_path, SAMPLE_TABLE, "current_side = ?", ("me",))
        print(f"Raw matches: {total_raw_matches}")
        print("Pending matches: 0")
        print("Valid newly expanded matches: 0")
        print("Skipped newly expanded matches: 0")
        print(f"All training samples in sqlite: {existing_all_count} (+0, replaced 0)")
        print(f"Me-turn training samples in sqlite: {existing_me_only_count} (+0, replaced 0)")
        return

    valid_match_count = 0
    skipped_match_count = 0
    total_samples = 0
    total_me_only = 0

    all_samples = []
    me_only_samples = []

    if not incremental and output_db_path and os.path.exists(output_db_path):
        os.remove(output_db_path)
    existing_all = load_pt(output_all_path, default=[]) if export_pt and incremental and output_all_path else []
    existing_me_only = load_pt(output_me_only_path, default=[]) if export_pt and incremental and output_me_only_path else []
    worker_count = int(max_workers or (os.cpu_count() or 4))
    worker_count = max(1, min(len(pending_matches), worker_count)) if pending_matches else 1

    def consume_result(result):
        nonlocal skipped_match_count, valid_match_count, total_samples, total_me_only
        if result is None:
            skipped_match_count += 1
            return

        samples, me_samples = result
        all_samples.extend(samples)
        me_only_samples.extend(me_samples)
        valid_match_count += 1
        total_samples += len(samples)
        total_me_only += len(me_samples)

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(expand_one_match, match) for match in pending_matches]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Expanding"):
                consume_result(future.result())
    except (PermissionError, OSError) as exc:
        print(f"[WARN] 进程池不可用，回退到串行展开: {exc}")
        for match in tqdm(pending_matches, desc="Expanding (serial)"):
            consume_result(expand_one_match(match))

    db_appended, db_replaced, db_skipped = upsert_record_rows(
        output_db_path,
        SAMPLE_TABLE,
        all_samples,
        SAMPLE_INDEX_COLUMNS,
        key_fn=lambda row: row.get("sample_id"),
    )

    if export_pt:
        print("Merging expanded samples with existing output...", flush=True)
        if pending_matches:
            all_samples = existing_all + all_samples
            me_only_samples = existing_me_only + me_only_samples
            all_appended, all_replaced = len(all_samples) - len(existing_all), 0
            me_appended, me_replaced = len(me_only_samples) - len(existing_me_only), 0
        else:
            all_samples, me_only_samples = existing_all, existing_me_only
            all_appended = all_replaced = 0
            me_appended = me_replaced = 0
        save_with_log("all samples pt", save_pt, all_samples, output_all_path)
        save_with_log("me-turn samples pt", save_pt, me_only_samples, output_me_only_path)
    updated_processed_match_ids = processed_match_ids | {raw_match_id(match) for match in pending_matches}
    stats = {
        "raw_matches": int(total_raw_matches),
        "pending_matches": int(len(pending_matches)),
        "valid_newly_expanded_matches": int(valid_match_count),
        "skipped_newly_expanded_matches": int(skipped_match_count),
        "all_training_samples": int(count_record_rows(output_db_path, SAMPLE_TABLE)),
        "me_turn_training_samples": int(count_record_rows(output_db_path, SAMPLE_TABLE, "current_side = ?", ("me",))),
    }
    save_manifest(manifest_path, updated_processed_match_ids, stats=stats)

    print(f"Raw matches: {total_raw_matches}")
    print(f"Pending matches: {len(pending_matches)}")
    print(f"Valid newly expanded matches: {valid_match_count}")
    print(f"Skipped newly expanded matches: {skipped_match_count}")
    print(f"SQLite samples: +{db_appended}, replaced {db_replaced}, skipped {db_skipped}")
    print(f"All training samples in sqlite: {stats['all_training_samples']}")
    print(f"Me-turn training samples in sqlite: {stats['me_turn_training_samples']}")
    if export_pt:
        print(f"All training samples pt: {len(all_samples)} (+{all_appended}, replaced {all_replaced})")
        print(f"Me-turn training samples pt: {len(me_only_samples)} (+{me_appended}, replaced {me_replaced})")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand raw BP matches into training samples.")
    parser.add_argument("--input-path", default=str(DEFAULT_RAW_DB_PATH))
    parser.add_argument("--output-db-path", default=str(DEFAULT_OUTPUT_DB_PATH))
    parser.add_argument("--output-all-path", default=str(DEFAULT_ALL_SAMPLES_PATH))
    parser.add_argument("--output-me-only-path", default=str(DEFAULT_ME_ONLY_SAMPLES_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--fetch-batch-size", type=int, default=DEFAULT_FETCH_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--export-pt", action="store_true")
    parser.add_argument("--incremental", dest="incremental", action="store_true", default=True)
    parser.add_argument("--force-rebuild", dest="incremental", action="store_false")
    args = parser.parse_args()
    expand_from_matches_raw(
        incremental=args.incremental,
        input_path=args.input_path,
        output_db_path=args.output_db_path,
        output_all_path=args.output_all_path,
        output_me_only_path=args.output_me_only_path,
        manifest_path=args.manifest_path,
        fetch_batch_size=args.fetch_batch_size,
        max_workers=args.max_workers,
        export_pt=args.export_pt,
    )
