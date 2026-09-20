import argparse
from collections import Counter

from bp_core.action_space import DraftTimeline
from bp_core.data_common import load_record_rows
from bp_core.relation_common import side_for_order
from bp_core.transformer_common import PROJECT_ROOT, normalize_name


def audit_samples(samples):
    issue_counts = Counter()
    examples = {}
    invalid_samples = 0
    for sample in samples:
        sample_invalid = False
        turn = int(sample.get("turn", 0) or 0)
        first_pick_side = sample.get("first_pick_side")
        if "events" in sample:
            timeline = DraftTimeline(sample.get("events") or [])
            my_picks = list(timeline.picks_for("me"))
            enemy_picks = list(timeline.picks_for("enemy"))
        else:
            my_picks = sample.get("my_picks_slugs") or sample.get("my_picks") or []
            enemy_picks = sample.get("enemy_picks_slugs") or sample.get("enemy_picks") or []
        current_side = normalize_name(sample.get("current_side"))
        expected_side = side_for_order(turn, first_pick_side)

        def issue(name):
            nonlocal sample_invalid
            issue_counts[name] += 1
            examples.setdefault(name, sample.get("sample_id") or sample.get("match_id"))
            sample_invalid = True

        if normalize_name(first_pick_side) not in {"me", "enemy"}:
            issue("invalid_first_pick_side")
        elif expected_side is None or current_side != expected_side:
            issue("current_side_does_not_match_turn")
        if len(my_picks) + len(enemy_picks) != max(0, turn - 1):
            issue("prefix_pick_count_does_not_match_turn")
        next_action = sample.get("next_action_slugs") or []
        next_pick = (next_action[0] if next_action else None) or sample.get("next_pick_slug") or sample.get("next_pick")
        if not next_pick:
            issue("missing_next_pick")
        elif normalize_name(next_pick) in {normalize_name(value) for value in my_picks + enemy_picks}:
            issue("next_pick_already_in_prefix")
        if sample_invalid:
            invalid_samples += 1

    return {
        "sample_count": len(samples),
        "issue_counts": dict(issue_counts),
        "issue_examples": examples,
        "valid_sample_count": len(samples) - invalid_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit chronological BP prefix samples.")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "bp_samples.sqlite"))
    args = parser.parse_args()
    samples = load_record_rows(args.data_path, "bp_samples")
    if not samples:
        print(f"no samples found: {args.data_path}")
        raise SystemExit(2)
    report = audit_samples(samples)
    print(report)
    if report["issue_counts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
