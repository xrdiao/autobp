from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bp_data_common import NameResolver, load_json, load_record_rows, normalize_relation_sample, upsert_record_rows

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "bp_samples.sqlite"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "bp_samples_normalized.sqlite"
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "json" / "monster_alias_map.json"
SAMPLE_INDEX_COLUMNS = [
    "sample_id",
    "match_id",
    "player_id",
    "current_side",
    "rank_tier",
    "dataset_version",
    "source_html",
]


def normalize_training_samples(
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    alias_path=DEFAULT_ALIAS_PATH,
):
    samples = load_record_rows(input_path, "bp_samples", "current_side = ?", ("me",))
    alias_map = load_json(alias_path, default={})
    resolver = NameResolver(alias_map=alias_map)
    normalized = [normalize_relation_sample(sample, resolver) for sample in samples]
    upsert_record_rows(
        output_path,
        "bp_samples",
        normalized,
        SAMPLE_INDEX_COLUMNS,
        key_fn=lambda row: row.get("sample_id"),
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    normalize_training_samples()
