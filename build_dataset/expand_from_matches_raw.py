import os
import argparse
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bp_core.data_common import count_record_rows, load_json, safe_pickle_loads, save_json, upsert_record_rows
from bp_core.bp_common import side_for_order
from bp_core.names import normalize_name

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DB_PATH = DATA_DIR / "matches_raw.sqlite"
DEFAULT_OUTPUT_DB_PATH = DATA_DIR / "bp_samples.sqlite"
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
                    match = safe_pickle_loads(record)
                except Exception as exc:
                    print(f"[WARN] failed to decode raw match {key}: {exc}", flush=True)
                    continue
                yield key, match
    finally:
        conn.close()


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
    first_side = combined_sequence[0].get("side")
    if first_side not in {"me", "enemy"}:
        return False
    valid_slots = all(
        item["order"] == i
        and item.get("side") == side_for_order(i, first_side)
        and (item.get("slug") or item.get("name"))
        for i, item in enumerate(combined_sequence, start=1)
    )
    if not valid_slots:
        return False
    identifiers = [normalize_name(item.get("slug") or item.get("name")) for item in combined_sequence]
    return all(identifiers) and len(set(identifiers)) == len(identifiers)


def build_pick_events(combined_sequence: list):
    """Group the chronological ten-slot sequence into BP events."""
    events = []
    cursor = 0
    while cursor < len(combined_sequence):
        first = combined_sequence[cursor]
        side = first["side"]
        group = [first]
        if cursor + 1 < len(combined_sequence) and combined_sequence[cursor + 1]["side"] == side:
            group.append(combined_sequence[cursor + 1])
        members = sorted(
            group,
            key=lambda item: normalize_name(item.get("slug") or item.get("name")),
        )
        events.append(
            {
                "order": len(events) + 1,
                "start_slot": group[0]["order"],
                "end_slot": group[-1]["order"],
                "side": side,
                "action_type": "pick",
                "monsters": [item.get("slug") or item.get("name") for item in members],
                "monster_names": [item.get("name") or item.get("slug") for item in members],
                "is_pair": len(group) == 2,
            }
        )
        cursor += len(group)
    return events


def prefix_events_for_slot(events, slot):
    """Return complete events before a slot and its full next action."""
    prefix = [
        {
            key: value
            for key, value in event.items()
            if key not in {"start_slot", "end_slot"}
        }
        for event in events
        if event["end_slot"] < slot
    ]
    next_event = next(
        (event for event in events if event["start_slot"] <= slot <= event["end_slot"]),
        None,
    )
    next_action = [] if next_event is None else list(next_event["monsters"])
    return prefix, next_action

def expand_one_match(match: dict):
    combined_sequence = build_combined_sequence(match)
    if not validate_match_sequence(combined_sequence):
        return None

    first_pick_side = combined_sequence[0]["side"]
    pick_events = build_pick_events(combined_sequence)

    my_picks_raw = match.get("my_picks", [])
    enemy_picks_raw = match.get("enemy_picks", [])
    my_leader_slug = get_side_value(my_picks_raw, "is_leader", "slug")
    enemy_leader_slug = get_side_value(enemy_picks_raw, "is_leader", "slug")

    # Keep both Ban directions. ``ban_slug`` remains the historical alias for
    # the current player's Ban target (an enemy monster).
    enemy_final_slugs = {
        normalize_name(item.get("slug") or item.get("name"))
        for item in enemy_picks_raw
        if item.get("slug") or item.get("name")
    }
    ban_slug = normalize_name(get_side_value(enemy_picks_raw, "is_banned", "slug"))
    if ban_slug not in enemy_final_slugs:
        ban_slug = ""
    my_final_slugs = {
        normalize_name(item.get("slug") or item.get("name"))
        for item in my_picks_raw
        if item.get("slug") or item.get("name")
    }
    enemy_ban_slug = normalize_name(get_side_value(my_picks_raw, "is_banned", "slug"))
    if enemy_ban_slug not in my_final_slugs:
        enemy_ban_slug = ""

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

    full_events = [
        {
            key: value
            for key, value in event.items()
            if key not in {"start_slot", "end_slot"}
        }
        for event in pick_events
    ]
    samples = []
    for event in pick_events:
        prefix_events, next_action_slugs = prefix_events_for_slot(
            pick_events,
            event["start_slot"],
        )
        sample = {
            "sample_id": f"{match_id}_{event['start_slot']}",
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
            "time_ago": time_ago,
            "source_html": source_html,
            "self_name": self_info.get("name"),
            "self_rating": self_info.get("rating"),
            "opponent_name": opponent_info.get("name"),
            "opponent_rating": opponent_info.get("rating"),
            "turn": event["start_slot"],
            "first_pick_side": first_pick_side,
            "current_side": event["side"],
            "events": prefix_events,
            "full_events": full_events,
            "next_action_slugs": next_action_slugs,
            "my_leader_slug": my_leader_slug,
            "enemy_leader_slug": enemy_leader_slug,
            "ban_slug": ban_slug,
            "my_ban_slug": ban_slug,
            "enemy_ban_slug": enemy_ban_slug,
        }
        samples.append(sample)

    return samples


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
    manifest_path=str(DEFAULT_MANIFEST_PATH),
    fetch_batch_size=DEFAULT_FETCH_BATCH_SIZE,
    max_workers=None,
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
    all_samples = []

    if not incremental and output_db_path and os.path.exists(output_db_path):
        os.remove(output_db_path)
    worker_count = int(max_workers or (os.cpu_count() or 4))
    worker_count = max(1, min(len(pending_matches), worker_count)) if pending_matches else 1

    def consume_result(result):
        nonlocal skipped_match_count, valid_match_count
        if result is None:
            skipped_match_count += 1
            return

        all_samples.extend(result)
        valid_match_count += 1

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(expand_one_match, match) for match in pending_matches]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Expanding"):
            consume_result(future.result())

    db_appended, db_replaced, db_skipped = upsert_record_rows(
        output_db_path,
        SAMPLE_TABLE,
        all_samples,
        SAMPLE_INDEX_COLUMNS,
        key_fn=lambda row: row.get("sample_id"),
    )

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
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand raw BP matches into training samples.")
    parser.add_argument("--input-path", default=str(DEFAULT_RAW_DB_PATH))
    parser.add_argument("--output-db-path", default=str(DEFAULT_OUTPUT_DB_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--fetch-batch-size", type=int, default=DEFAULT_FETCH_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--incremental", dest="incremental", action="store_true", default=True)
    parser.add_argument("--force-rebuild", dest="incremental", action="store_false")
    args = parser.parse_args()
    expand_from_matches_raw(
        incremental=args.incremental,
        input_path=args.input_path,
        output_db_path=args.output_db_path,
        manifest_path=args.manifest_path,
        fetch_batch_size=args.fetch_batch_size,
        max_workers=args.max_workers,
    )
