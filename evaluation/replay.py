from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from bp_core.data_common import load_record_rows
from bp_core.relation_common import infer_pick_count, side_for_order
from bp_recommender.relation import BPRelationRecommender
from bp_core.transformer_common import PROJECT_ROOT, normalize_name


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bp_samples.sqlite"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "models" / "bp_relation_split_manifest.json"


def _sample_slugs(sample, side):
    values = sample.get(f"{side}_picks_slugs")
    if values is None:
        values = sample.get(f"{side}_picks") or []
    return [str(value) for value in values if value]


def _next_pick_slug(sample, relation_model):
    value = sample.get("next_pick_slug") or sample.get("next_pick")
    return relation_model._to_slug(value)


def _candidate_pool(relation_model):
    return sorted(
        set(relation_model._known_model_slugs)
        | set(relation_model.lucksack_stats.candidate_slugs())
    )


def _plan_slugs(plan, relation_model):
    return {
        relation_model._to_slug(value)
        for value in (plan.get("plan") or plan.get("monsters") or [])
        if value
    }


def _result_label(sample):
    value = normalize_name(sample.get("result"))
    if value in {"win", "victory", "winner"}:
        return 1.0
    if value in {"lose", "loss", "defeat"}:
        return 0.0
    value = sample.get("won")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
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
            f"match split manifest is required for --eval-split {split}: {manifest_path}"
        )
    selected = []
    for sample in samples:
        group = groups.get(f"match_id:{_match_id(sample)}")
        if group and group.get("split") == split:
            selected.append(sample)
    return selected


def _select_match_representatives(samples, seed=42):
    groups = defaultdict(list)
    for sample in samples:
        groups[_match_id(sample)].append(sample)
    rng = random.Random(int(seed))
    selected = []
    for match_id in sorted(groups):
        rows = sorted(
            groups[match_id],
            key=lambda sample: str(sample.get("sample_id") or ""),
        )
        if rows:
            selected.append(rng.choice(rows))
    selected.sort(key=lambda sample: str(sample.get("sample_id") or ""))
    return selected


def _binary_metrics(records):
    records = list(records or [])
    if not records:
        return {
            "evaluated": 0,
            "brier_score": None,
            "log_loss": None,
            "ece": None,
            "mean_predicted_win_rate": None,
            "actual_win_rate": None,
        }
    probabilities = [min(1.0 - 1e-6, max(1e-6, float(row["probability"]))) for row in records]
    labels = [float(row["label"]) for row in records]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(records)
    log_loss = sum(
        -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
        for probability, label in zip(probabilities, labels)
    ) / len(records)
    ece = 0.0
    for bucket in range(10):
        lower = bucket / 10.0
        upper = (bucket + 1) / 10.0
        rows = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if lower <= probability < upper
            or (bucket == 9 and lower <= probability <= upper)
        ]
        if rows:
            ece += len(rows) / len(records) * abs(
                sum(probability for probability, _label in rows) / len(rows)
                - sum(label for _probability, label in rows) / len(rows)
            )
    return {
        "evaluated": len(records),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 6),
        "mean_predicted_win_rate": round(sum(probabilities) / len(probabilities), 6),
        "actual_win_rate": round(sum(labels) / len(labels), 6),
    }


def _group_binary_metrics(records, field):
    groups = defaultdict(list)
    for row in records:
        groups[str(row.get(field) or "unknown")].append(row)
    return {
        key: _binary_metrics(rows)
        for key, rows in sorted(groups.items())
    }


def _select_samples(samples, limit, seed=42):
    selected = [
        sample
        for sample in samples
        if normalize_name(sample.get("current_side")) == "me"
        and (sample.get("next_pick_slug") or sample.get("next_pick"))
        and normalize_name(sample.get("first_pick_side")) in {"me", "enemy"}
    ]
    selected.sort(key=lambda sample: str(sample.get("sample_id") or ""))
    if limit > 0:
        selected = random.Random(int(seed)).sample(
            selected,
            min(int(limit), len(selected)),
        )
        selected.sort(key=lambda sample: str(sample.get("sample_id") or ""))
    return selected


def evaluate_replay(
    relation_model,
    samples,
    prefix_model=None,
    top_k=6,
    max_pair_candidates=400,
    outcome_samples=None,
    seed=42,
):
    candidate_pool = _candidate_pool(relation_model)
    rows = []
    hit_counts = defaultdict(int)
    turn_counts = defaultdict(int)
    total_latency = 0.0
    value_rows = []
    value_refs = []

    rerank_my_plans = None
    if prefix_model is not None:
        from app_web import rerank_my_plans

    for sample in samples:
        my_picks = _sample_slugs(sample, "my")
        enemy_picks = _sample_slugs(sample, "enemy")
        first_pick_side = normalize_name(sample.get("first_pick_side"))
        turn = int(sample.get("turn") or len(my_picks) + len(enemy_picks) + 1)
        pick_count = infer_pick_count(my_picks, enemy_picks, first_pick_side)
        if pick_count <= 0:
            continue

        started = time.perf_counter()
        plans = relation_model.recommend_pick_plans(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            pick_count=pick_count,
            turn=turn,
            first_pick_side=first_pick_side,
            candidate_pool=candidate_pool,
            use_my_box_for_reco=False,
            top_k=max(int(top_k), 3),
            max_pair_candidates=max_pair_candidates,
            preserve_evidence=True,
        )
        relation_plans = list(plans)
        if prefix_model is not None and plans:
            plans = rerank_my_plans(
                plans,
                prefix_model,
                relation_model,
                {
                    "my_picks": my_picks,
                    "enemy_picks": enemy_picks,
                    "turn": turn,
                    "first_pick_side": first_pick_side,
                    "current_side": side_for_order(turn, first_pick_side) or "me",
                    "my_leader": sample.get("my_leader") or "",
                    "enemy_leader": sample.get("enemy_leader") or "",
                    "my_box": [],
                    "use_my_box": False,
                    "max_pair_candidates": max_pair_candidates,
                },
                {"options": candidate_pool},
                top_k,
            )
        elapsed = time.perf_counter() - started
        total_latency += elapsed

        actual = _next_pick_slug(sample, relation_model)
        hit_row = {
            "sample_id": sample.get("sample_id"),
            "turn": turn,
            "pick_count": pick_count,
            "actual_next_pick": actual,
            "top_plans": [list(_plan_slugs(plan, relation_model)) for plan in plans[:top_k]],
            "latency_ms": round(elapsed * 1000.0, 3),
        }
        if prefix_model is not None and plans:
            recommended = _plan_slugs(plans[0], relation_model)
            relation_top = _plan_slugs(relation_plans[0], relation_model)
            recommended_state = {
                "my_picks": my_picks + list(recommended),
                "enemy_picks": enemy_picks,
                "turn": turn + len(recommended),
                "first_pick_side": first_pick_side,
                "current_side": side_for_order(
                    turn + len(recommended),
                    first_pick_side,
                ) or "enemy",
                "my_leader": sample.get("my_leader") or "",
                "enemy_leader": sample.get("enemy_leader") or "",
            }
            actual_state = {
                "my_picks": my_picks + [actual],
                "enemy_picks": enemy_picks,
                "turn": turn + 1,
                "first_pick_side": first_pick_side,
                "current_side": side_for_order(turn + 1, first_pick_side) or "enemy",
                "my_leader": sample.get("my_leader") or "",
                "enemy_leader": sample.get("enemy_leader") or "",
            }
            relation_state = {
                "my_picks": my_picks + list(relation_top),
                "enemy_picks": enemy_picks,
                "turn": turn + len(relation_top),
                "first_pick_side": first_pick_side,
                "current_side": side_for_order(
                    turn + len(relation_top),
                    first_pick_side,
                ) or "enemy",
                "my_leader": sample.get("my_leader") or "",
                "enemy_leader": sample.get("enemy_leader") or "",
            }
            value_refs.append(len(rows))
            value_rows.extend((recommended_state, actual_state, relation_state))
        for rank in (1, 3):
            hit = any(actual in _plan_slugs(plan, relation_model) for plan in plans[:rank])
            hit_counts[f"top{rank}"] += int(hit)
            hit_row[f"top{rank}"] = bool(hit)
        turn_counts[turn] += 1
        rows.append(hit_row)

    if prefix_model is not None and value_rows:
        value_evaluations = prefix_model.evaluate_states(value_rows, return_detail=True)
        for row_index, offset in zip(value_refs, range(0, len(value_evaluations), 3)):
            recommended_value = float(value_evaluations[offset]["calibrated_win_rate"])
            actual_value = float(value_evaluations[offset + 1]["calibrated_win_rate"])
            relation_value = float(value_evaluations[offset + 2]["calibrated_win_rate"])
            rows[row_index]["recommended_value"] = round(recommended_value, 6)
            rows[row_index]["actual_value"] = round(actual_value, 6)
            rows[row_index]["relation_value"] = round(relation_value, 6)
            rows[row_index]["value_lift"] = round(recommended_value - actual_value, 6)
            rows[row_index]["search_lift"] = round(recommended_value - relation_value, 6)

    outcome_eval = None
    if prefix_model is not None:
        objective_samples = list(
            outcome_samples
            if outcome_samples is not None
            else _select_match_representatives(samples, seed=seed)
        )
        outcome_states = []
        outcome_meta = []
        for sample in objective_samples:
            label = _result_label(sample)
            if label is None:
                continue
            my_picks = _sample_slugs(sample, "my")
            enemy_picks = _sample_slugs(sample, "enemy")
            first_pick_side = normalize_name(sample.get("first_pick_side"))
            turn = int(sample.get("turn") or len(my_picks) + len(enemy_picks) + 1)
            outcome_states.append(
                {
                    "my_picks": my_picks,
                    "enemy_picks": enemy_picks,
                    "turn": turn,
                    "first_pick_side": first_pick_side,
                    "current_side": side_for_order(turn, first_pick_side) or "me",
                    "my_leader": sample.get("my_leader") or "",
                    "enemy_leader": sample.get("enemy_leader") or "",
                }
            )
            outcome_meta.append(
                {
                    "label": label,
                    "turn": turn,
                    "rank_tier": sample.get("rank_tier") or "unknown",
                    "match_id": _match_id(sample),
                }
            )

        outcome_records = []
        for start in range(0, len(outcome_states), 512):
            evaluations = prefix_model.evaluate_states(
                outcome_states[start : start + 512],
                return_detail=True,
            )
            for meta, evaluation in zip(
                outcome_meta[start : start + 512],
                evaluations,
            ):
                outcome_records.append(
                    {
                        **meta,
                        "probability": float(
                            evaluation.get("calibrated_win_rate", 0.5)
                        ),
                    }
                )
        outcome_eval = _binary_metrics(outcome_records)
        outcome_eval.update(
            {
                "unique_matches": len({_match_id(row) for row in outcome_records}),
                "sampling": "one_prefix_per_match",
                "by_turn": _group_binary_metrics(outcome_records, "turn"),
                "by_rank": _group_binary_metrics(outcome_records, "rank_tier"),
            }
        )

    evaluated = len(rows)
    summary = {
        "evaluated": evaluated,
        "top1_hit_rate": hit_counts["top1"] / evaluated if evaluated else 0.0,
        "top3_hit_rate": hit_counts["top3"] / evaluated if evaluated else 0.0,
        "mean_latency_ms": (total_latency * 1000.0 / evaluated) if evaluated else 0.0,
        "mean_recommended_value": (
            sum(row["recommended_value"] for row in rows if "recommended_value" in row)
            / max(1, sum("recommended_value" in row for row in rows))
            if rows else None
        ),
        "mean_actual_value": (
            sum(row["actual_value"] for row in rows if "actual_value" in row)
            / max(1, sum("actual_value" in row for row in rows))
            if rows else None
        ),
        "mean_value_lift": (
            sum(row["value_lift"] for row in rows if "value_lift" in row)
            / max(1, sum("value_lift" in row for row in rows))
            if rows else None
        ),
        "recommendation_beats_actual_rate": (
            sum(row["value_lift"] > 0 for row in rows if "value_lift" in row)
            / max(1, sum("value_lift" in row for row in rows))
            if rows else None
        ),
        "mean_relation_value": (
            sum(row["relation_value"] for row in rows if "relation_value" in row)
            / max(1, sum("relation_value" in row for row in rows))
            if rows else None
        ),
        "mean_search_lift": (
            sum(row["search_lift"] for row in rows if "search_lift" in row)
            / max(1, sum("search_lift" in row for row in rows))
            if rows else None
        ),
        "search_beats_relation_rate": (
            sum(row["search_lift"] > 0 for row in rows if "search_lift" in row)
            / max(1, sum("search_lift" in row for row in rows))
            if rows else None
        ),
        "by_turn": {
            str(turn): {
                "evaluated": count,
                "top1_hit_rate": sum(
                    1 for row in rows if row["turn"] == turn and row["top1"]
                ) / count,
                "top3_hit_rate": sum(
                    1 for row in rows if row["turn"] == turn and row["top3"]
                ) / count,
            }
            for turn, count in sorted(turn_counts.items())
        },
        "outcome_eval": outcome_eval,
        "rows": rows,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Replay historical RTA BP prefixes and evaluate recommendation hits.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--eval-split",
        choices=("val", "train", "all"),
        default="val",
        help="Match-level split to evaluate; val avoids overlap with relation training.",
    )
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    parser.add_argument("--limit", type=int, default=2000, help="Number of me-turn prefixes; 0 evaluates all.")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed; used when --limit > 0.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-pair-candidates", type=int, default=400)
    parser.add_argument("--with-value", action="store_true", help="Also rerank with the Prefix Value model.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_record_rows(Path(args.data_path), "bp_samples")
    samples = _filter_eval_split(
        samples,
        args.eval_split,
        args.split_manifest,
    )
    samples = _select_samples(samples, int(args.limit), seed=int(args.seed))
    relation_model = BPRelationRecommender()
    prefix_model = None
    if args.with_value:
        from bp_recommender.prefix_value import BPPrefixValueRecommender

        prefix_model = BPPrefixValueRecommender()
    summary = evaluate_replay(
        relation_model,
        samples,
        prefix_model=prefix_model,
        top_k=args.top_k,
        max_pair_candidates=args.max_pair_candidates,
        seed=int(args.seed),
    )
    output = {
        "data_path": str(args.data_path),
        "eval_split": args.eval_split,
        "split_manifest": str(args.split_manifest),
        "candidate_matches": len({_match_id(sample) for sample in samples}),
        "with_value": bool(args.with_value),
        "sample_limit": int(args.limit),
        "sampling_strategy": "seeded_random" if int(args.limit) > 0 else "all",
        "sampling_seed": int(args.seed),
        "top_k": int(args.top_k),
        "max_pair_candidates": int(args.max_pair_candidates),
        **{key: value for key, value in summary.items() if key != "rows"},
    }
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
