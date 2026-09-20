"""Replay evaluation for the legacy specialized BP models on event states."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from bp_core.action_space import DraftTimeline, transition_state, validate_timeline
from bp_core.data_common import load_record_rows
from bp_core.bp_common import infer_pick_count, side_for_order
from bp_core.names import PROJECT_ROOT, normalize_name
from bp_recommender.legacy_adapter import LegacyBPRecommender


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bp_samples.sqlite"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "models" / "bp_shared_split_manifest.json"
def _canonical_action(values):
    result, seen = [], set()
    for value in values or ():
        value = str(value or "").strip()
        key = normalize_name(value)
        if value and key and key not in seen:
            result.append(value)
            seen.add(key)
    if len(result) > 1:
        result.sort(key=lambda value: (normalize_name(value), value.casefold(), value))
    return tuple(result)


def _sample_slugs(sample, side):
    timeline = DraftTimeline(sample["events"])
    return [str(value) for value in timeline.picks_for(side) if value]


def _to_slug(model, value):
    return model._to_slug(value) if value else ""


def _actual_action_slugs(sample, model, pick_count):
    values = [_to_slug(model, value) for value in sample["next_action_slugs"]]
    values = [value for value in values if value]
    if len(values) != int(pick_count):
        return ()
    return _canonical_action(values)


def _candidate_pool(model):
    return sorted(
        set(model.known_slugs)
        | set(getattr(model, "external_slugs", ()))
    )


def _plan_action(plan, model):
    return _canonical_action(_to_slug(model, value) for value in plan["plan"])


def _result_label(sample):
    value = normalize_name(sample.get("result"))
    if value in {"win", "victory", "winner"}:
        return 1.0
    if value in {"lose", "loss", "defeat"}:
        return 0.0
    return None


def _match_id(sample):
    return str(sample.get("match_id") or sample.get("sample_id") or "")


def _load_split_manifest(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("groups") or {}


def _filter_eval_split(samples, split, manifest_path):
    if split == "all":
        return list(samples)
    groups = _load_split_manifest(manifest_path)
    if not groups:
        raise FileNotFoundError(
            f"BP split manifest is required for --eval-split {split}: {manifest_path}"
        )
    filtered = [
        sample
        for sample in samples
        if (
            groups.get(_match_id(sample))
            or groups.get(f"match_id:{_match_id(sample)}")
            or groups.get(f"sample_id:{_match_id(sample)}")
            or {}
        ).get("split") == split
    ]
    if samples and not filtered:
        raise ValueError(
            f"split manifest matched no samples for --eval-split {split}; "
            "check manifest group keys against sample match_id/sample_id"
        )
    return filtered


def _select_match_representatives(samples, seed=42):
    groups = defaultdict(list)
    for sample in samples:
        groups[_match_id(sample)].append(sample)
    rng = random.Random(int(seed))
    selected = []
    for match_id in sorted(groups):
        rows = sorted(groups[match_id], key=lambda row: str(row.get("sample_id") or ""))
        if rows:
            selected.append(rng.choice(rows))
    return sorted(selected, key=lambda row: str(row.get("sample_id") or ""))


def _binary_metrics(records):
    records = list(records or [])
    if not records:
        return {"evaluated": 0, "brier_score": None, "log_loss": None, "ece": None,
                "mean_predicted_win_rate": None, "actual_win_rate": None}
    probabilities = [min(1.0 - 1e-6, max(1e-6, float(row["probability"]))) for row in records]
    labels = [float(row["label"]) for row in records]
    brier = sum((p - label) ** 2 for p, label in zip(probabilities, labels)) / len(records)
    log_loss = sum(
        -(label * math.log(p) + (1.0 - label) * math.log(1.0 - p))
        for p, label in zip(probabilities, labels)
    ) / len(records)
    ece = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        rows = [(p, label) for p, label in zip(probabilities, labels)
                if lower <= p < upper or (bucket == 9 and lower <= p <= upper)]
        if rows:
            ece += len(rows) / len(records) * abs(
                sum(p for p, _ in rows) / len(rows) - sum(label for _, label in rows) / len(rows)
            )
    return {"evaluated": len(records), "brier_score": round(brier, 6),
            "log_loss": round(log_loss, 6), "ece": round(ece, 6),
            "mean_predicted_win_rate": round(sum(probabilities) / len(probabilities), 6),
            "actual_win_rate": round(sum(labels) / len(labels), 6)}


def _group_binary_metrics(records, field):
    groups = defaultdict(list)
    for row in records:
        groups[str(row.get(field) or "unknown")].append(row)
    return {key: _binary_metrics(rows) for key, rows in sorted(groups.items())}


def _select_samples(samples, limit, seed=42):
    selected = []
    for sample in samples:
        if normalize_name(sample.get("current_side")) != "me":
            continue
        if normalize_name(sample.get("first_pick_side")) not in {"me", "enemy"}:
            continue
        try:
            timeline = DraftTimeline(sample["events"])
            turn = int(sample["turn"])
            next_action = list(sample["next_action_slugs"])
        except (KeyError, TypeError, ValueError):
            continue
        if timeline.slots_used != turn - 1 or len(next_action) not in {1, 2}:
            continue
        my_picks = list(timeline.picks_for("me"))
        enemy_picks = list(timeline.picks_for("enemy"))
        expected = infer_pick_count(my_picks, enemy_picks, sample["first_pick_side"])
        if len(next_action) != expected:
            continue
        selected.append(sample)
    selected.sort(key=lambda row: str(row.get("sample_id") or ""))
    if limit > 0:
        selected = random.Random(int(seed)).sample(selected, min(int(limit), len(selected)))
        selected.sort(key=lambda row: str(row.get("sample_id") or ""))
    return selected


def _state_after_action(sample, action):
    timeline = DraftTimeline(sample["events"])
    turn = int(sample["turn"])
    first_pick_side = normalize_name(sample["first_pick_side"])
    validate_timeline(timeline, first_pick_side)
    if timeline.slots_used != turn - 1:
        raise ValueError("sample event prefix does not match turn")
    state = {
        "my_picks": list(timeline.picks_for("me")),
        "enemy_picks": list(timeline.picks_for("enemy")),
        "turn": timeline.slots_used + 1,
        "first_pick_side": first_pick_side,
        "current_side": side_for_order(timeline.slots_used + 1, first_pick_side),
        "events": timeline.to_dict(),
        "my_leader": sample.get("my_leader_slug") or "",
        "enemy_leader": sample.get("enemy_leader_slug") or "",
    }
    return transition_state(
        state,
        {"side": "me", "action_type": "pick", "monsters": list(action)},
    )


def _plan_values(model, states):
    """Evaluate resulting states with the shared value head."""
    if not states:
        return []
    evaluations = model.evaluate_states(states, return_detail=True)
    return [float(row["calibrated_win_rate"]) for row in evaluations]


def evaluate_replay(shared_model, samples, top_k=6, max_pair_candidates=400,
                    outcome_samples=None, seed=42):
    candidate_pool = _candidate_pool(shared_model)
    rows, value_pairs = [], []
    hit_counts, turn_counts = defaultdict(int), defaultdict(int)
    total_latency = 0.0
    for sample in samples:
        my_picks, enemy_picks = _sample_slugs(sample, "me"), _sample_slugs(sample, "enemy")
        first_pick_side = normalize_name(sample["first_pick_side"])
        turn = int(sample["turn"])
        if DraftTimeline(sample["events"]).slots_used != turn - 1:
            continue
        pick_count = infer_pick_count(my_picks, enemy_picks, first_pick_side)
        actual_action = _actual_action_slugs(sample, shared_model, pick_count)
        if pick_count <= 0 or len(actual_action) != pick_count:
            continue
        started = time.perf_counter()
        plans = shared_model.recommend_pick_plans(
            my_picks=my_picks, enemy_picks=enemy_picks, pick_count=pick_count,
            turn=turn, first_pick_side=first_pick_side, candidate_pool=candidate_pool,
            use_my_box_for_reco=False, top_k=max(int(top_k), 3),
            max_pair_candidates=max_pair_candidates, events=sample["events"],
        )
        elapsed = time.perf_counter() - started
        total_latency += elapsed
        actions = [_plan_action(plan, shared_model) for plan in (plans or [])]
        actions = [action for action in actions if len(action) == pick_count]
        top_actions = actions[: int(top_k)]
        top_value = None
        if top_actions:
            top_value = _plan_values(
                shared_model,
                [_state_after_action(sample, top_actions[0])],
            )[0]
        actual_value = _plan_values(
            shared_model,
            [_state_after_action(sample, actual_action)],
        )[0]
        hit_row = {
            "sample_id": sample.get("sample_id"), "turn": turn,
            "pick_count": pick_count, "actual_next_pick": actual_action[0],
            "actual_action": list(actual_action), "top_plans": [list(a) for a in top_actions],
            "latency_ms": round(elapsed * 1000.0, 3),
        }
        actual_key = frozenset(actual_action)
        for rank in (1, 3):
            hit = any(len(action) == pick_count and frozenset(action) == actual_key
                      for action in actions[:rank])
            hit_counts[f"top{rank}"] += int(hit)
            hit_row[f"top{rank}"] = bool(hit)
        if top_value is not None:
            hit_row.update({"recommended_value": round(top_value, 6),
                            "actual_value": round(actual_value, 6),
                            "value_lift": round(top_value - actual_value, 6)})
            value_pairs.append(hit_row)
        turn_counts[turn] += 1
        rows.append(hit_row)

    objective_samples = list(outcome_samples if outcome_samples is not None
                             else _select_match_representatives(samples, seed=seed))
    outcome_states, outcome_meta = [], []
    for sample in objective_samples:
        label = _result_label(sample)
        if label is None:
            continue
        first_pick_side = normalize_name(sample["first_pick_side"])
        timeline = DraftTimeline(sample["events"])
        turn = int(sample["turn"])
        validate_timeline(timeline, first_pick_side)
        if timeline.slots_used != turn - 1:
            continue
        outcome_states.append({"my_picks": list(timeline.picks_for("me")),
                               "enemy_picks": list(timeline.picks_for("enemy")),
                               "turn": timeline.slots_used + 1,
                               "first_pick_side": first_pick_side,
                               "current_side": side_for_order(timeline.slots_used + 1, first_pick_side),
                               "events": timeline.to_dict(),
                               "my_leader": sample.get("my_leader_slug") or "",
                               "enemy_leader": sample.get("enemy_leader_slug") or ""})
        outcome_meta.append({"label": label, "turn": turn,
                             "rank_tier": sample.get("rank_tier") or "unknown",
                             "match_id": _match_id(sample)})
    outcome_records = []
    for start in range(0, len(outcome_states), 512):
        values = _plan_values(shared_model, outcome_states[start:start + 512])
        outcome_records.extend({**meta, "probability": value}
                               for meta, value in zip(outcome_meta[start:start + 512], values))
    outcome_eval = _binary_metrics(outcome_records)
    outcome_eval.update({"unique_matches": len({_match_id(row) for row in outcome_records}),
                         "sampling": "one_state_per_match",
                         "by_turn": _group_binary_metrics(outcome_records, "turn"),
                         "by_rank": _group_binary_metrics(outcome_records, "rank_tier")})

    evaluated, value_count = len(rows), len(value_pairs)
    summary = {
        "evaluated": evaluated,
        "top1_hit_rate": hit_counts["top1"] / evaluated if evaluated else 0.0,
        "top3_hit_rate": hit_counts["top3"] / evaluated if evaluated else 0.0,
        "mean_latency_ms": total_latency * 1000.0 / evaluated if evaluated else 0.0,
        "value_comparison_evaluated": value_count,
        "mean_recommended_value": sum(r["recommended_value"] for r in value_pairs) / value_count if value_count else None,
        "mean_actual_value": sum(r["actual_value"] for r in value_pairs) / value_count if value_count else None,
        "mean_value_lift": sum(r["value_lift"] for r in value_pairs) / value_count if value_count else None,
        "recommendation_beats_actual_rate": sum(r["value_lift"] > 0 for r in value_pairs) / value_count if value_count else None,
        "by_turn": {str(turn): {"evaluated": count,
                                 "top1_hit_rate": sum(1 for r in rows if r["turn"] == turn and r["top1"]) / count,
                                 "top3_hit_rate": sum(1 for r in rows if r["turn"] == turn and r["top3"]) / count}
                    for turn, count in sorted(turn_counts.items())},
        "outcome_eval": outcome_eval,
        "rows": rows,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Replay historical BP states with the legacy BP models.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--eval-split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-pair-candidates", type=int, default=400)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_record_rows(Path(args.data_path), "bp_samples")
    samples = _filter_eval_split(samples, args.eval_split, args.split_manifest)
    samples = _select_samples(samples, int(args.limit), seed=int(args.seed))
    if not samples:
        raise RuntimeError(
            f"no evaluable BP samples remain for --eval-split {args.eval_split}; "
            "check the dataset, split manifest, and event schema"
        )
    shared_model = LegacyBPRecommender()
    summary = evaluate_replay(shared_model, samples, top_k=args.top_k,
                               max_pair_candidates=args.max_pair_candidates, seed=int(args.seed))
    output = {"data_path": str(args.data_path), "eval_split": args.eval_split,
              "split_manifest": str(args.split_manifest),
              "candidate_matches": len({_match_id(sample) for sample in samples}),
              "model": "legacy_relation_prefix_ban", "sample_limit": int(args.limit),
              "sampling_strategy": "seeded_random" if int(args.limit) > 0 else "all",
              "sampling_seed": int(args.seed), "top_k": int(args.top_k),
              "max_pair_candidates": int(args.max_pair_candidates),
              **{key: value for key, value in summary.items() if key != "rows"}}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_output = dict(output)
        saved_output["rows"] = summary["rows"]
        path.write_text(json.dumps(saved_output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved replay rows: {path}")


if __name__ == "__main__":
    main()
