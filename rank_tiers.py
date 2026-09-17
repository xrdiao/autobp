from __future__ import annotations

from collections import Counter


RANK_TIER_SPECS = [
    {
        "key": "three_red",
        "label": "Three Red",
        "rank_min": 1,
        "rank_max": 300,
        "rank_weight": 1.25,
    },
    {
        "key": "two_red",
        "label": "Two Red",
        "rank_min": 301,
        "rank_max": 1000,
        "rank_weight": 1.15,
    },
    {
        "key": "one_red",
        "label": "One Red",
        "rank_min": 1001,
        "rank_max": 2500,
        "rank_weight": 1.05,
    },
    {
        "key": "three_green",
        "label": "Three Green",
        "rank_min": 2501,
        "rank_max": 5000,
        "rank_weight": 0.95,
    },
    {
        "key": "other",
        "label": "Other",
        "rank_min": 5001,
        "rank_max": None,
        "rank_weight": 0.85,
    },
]

RANK_TIER_WEIGHTS = {spec["key"]: float(spec["rank_weight"]) for spec in RANK_TIER_SPECS}
RANK_TIER_TARGET_PLAYERS = {
    "three_red": 250,
    "two_red": 250,
    "one_red": 250,
    "three_green": 250,
    "other": 250,
}
MAX_DEFAULT_RANK_TO_SCAN = 8000


def _safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def get_rank_tier_spec(rank=None, tier_key=None):
    normalized_key = str(tier_key or "").strip().lower()
    for spec in RANK_TIER_SPECS:
        if normalized_key and spec["key"] == normalized_key:
            return dict(spec)

    rank_value = _safe_int(rank)
    if rank_value is not None:
        for spec in RANK_TIER_SPECS:
            rank_min = int(spec["rank_min"])
            rank_max = spec["rank_max"]
            if rank_value >= rank_min and (rank_max is None or rank_value <= int(rank_max)):
                return dict(spec)

    return dict(RANK_TIER_SPECS[-1])


def enrich_rank_context(rank=None, tier_key=None, explicit_weight=None):
    spec = get_rank_tier_spec(rank=rank, tier_key=tier_key)
    rank_value = _safe_int(rank)
    try:
        rank_weight = float(explicit_weight) if explicit_weight is not None else float(spec["rank_weight"])
    except Exception:
        rank_weight = float(spec["rank_weight"])

    return {
        "player_rank": rank_value,
        "rank_tier": spec["key"],
        "rank_tier_label": spec["label"],
        "rank_weight": rank_weight,
        "sample_weight": rank_weight,
    }


def get_rank_context_from_record(record):
    record = record or {}
    explicit_weight = record.get("sample_weight")
    if explicit_weight in (None, ""):
        explicit_weight = record.get("rank_weight")
    return enrich_rank_context(
        rank=record.get("player_rank"),
        tier_key=record.get("rank_tier"),
        explicit_weight=explicit_weight,
    )


def summarize_rank_tier_counts(records):
    counter = Counter()
    for record in records or []:
        context = get_rank_context_from_record(record)
        counter[context["rank_tier"]] += 1
    return {spec["key"]: int(counter.get(spec["key"], 0)) for spec in RANK_TIER_SPECS}


def tier_targets_met(counter, targets=None):
    targets = targets or RANK_TIER_TARGET_PLAYERS
    for key, target in targets.items():
        if target is None:
            continue
        if int(counter.get(key, 0)) < int(target):
            return False
    return True
