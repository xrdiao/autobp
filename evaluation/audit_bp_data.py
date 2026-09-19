"""Audit the canonical event-level BP sample schema."""

from __future__ import annotations

import argparse
from collections import Counter

from bp_core.action_space import DraftTimeline
from bp_core.bp_common import infer_pick_count_for_side, side_for_order
from bp_core.data_common import load_record_rows
from bp_core.names import PROJECT_ROOT, normalize_name


REQUIRED_FIELDS = (
    "events",
    "full_events",
    "next_action_slugs",
    "turn",
    "first_pick_side",
    "current_side",
)


def _event_signature(event):
    return event.side, event.action_type, tuple(normalize_name(value) for value in event.monsters)


def _next_event(full_timeline, turn):
    slots = 0
    for event in full_timeline.events:
        if slots + 1 == int(turn):
            return event
        slots += event.count
    return None


def audit_samples(samples):
    issue_counts = Counter()
    examples = {}
    invalid_samples = 0

    for sample in samples:
        sample_invalid = False

        def issue(name):
            nonlocal sample_invalid
            issue_counts[name] += 1
            examples.setdefault(name, sample.get("sample_id") or sample.get("match_id"))
            sample_invalid = True

        missing = [key for key in REQUIRED_FIELDS if key not in sample]
        if missing:
            for key in missing:
                issue(f"missing_{key}")
            invalid_samples += 1
            continue

        first_pick_side = normalize_name(sample["first_pick_side"])
        current_side = normalize_name(sample["current_side"])
        if first_pick_side not in {"me", "enemy"}:
            issue("invalid_first_pick_side")
        if current_side not in {"me", "enemy"}:
            issue("invalid_current_side")

        try:
            turn = int(sample["turn"])
        except (TypeError, ValueError):
            turn = 0
            issue("invalid_turn")
        if turn < 1 or turn > 10:
            issue("turn_out_of_range")

        try:
            timeline = DraftTimeline(sample["events"])
            full_timeline = DraftTimeline(sample["full_events"])
        except (TypeError, ValueError):
            timeline = None
            full_timeline = None
            issue("missing_or_invalid_events")

        if timeline is not None and full_timeline is not None:
            if full_timeline.slots_used != 10:
                issue("full_events_not_ten_slots")
            if timeline.slots_used != turn - 1:
                issue("prefix_slots_do_not_match_turn")
            if len(timeline.events) >= len(full_timeline.events):
                issue("events_not_strict_prefix")
            for prefix_event, full_event in zip(timeline.events, full_timeline.events):
                if _event_signature(prefix_event) != _event_signature(full_event):
                    issue("events_not_full_prefix")
                    break

            slot = 1
            expected_sizes = []
            for event in full_timeline.events:
                expected_sizes.append(event.count)
                if event.action_type != "pick":
                    issue("full_events_contains_non_pick")
                for _monster in event.monsters:
                    if side_for_order(slot, first_pick_side) != event.side:
                        issue("full_events_side_pattern_mismatch")
                    slot += 1
            if expected_sizes != [1, 2, 2, 2, 2, 1]:
                issue("full_events_event_sizes_mismatch")

            expected_side = side_for_order(turn, first_pick_side) if 1 <= turn <= 10 else None
            if expected_side != current_side:
                issue("current_side_does_not_match_turn")

            my_picks = list(timeline.picks_for("me"))
            enemy_picks = list(timeline.picks_for("enemy"))
            all_picks = [normalize_name(value) for value in full_timeline.picks_for("me") + full_timeline.picks_for("enemy")]
            if len(all_picks) != 10 or len(set(all_picks)) != 10:
                issue("full_events_contains_duplicate_or_empty_monster")
            expected_count = infer_pick_count_for_side(
                my_picks,
                enemy_picks,
                first_pick_side,
                current_side,
            )
            next_action = sample["next_action_slugs"]
            if not isinstance(next_action, (list, tuple)):
                issue("next_action_is_not_a_list")
                next_action = []
            next_action = [normalize_name(value) for value in next_action]
            if len(next_action) == 2:
                next_action.sort()
            if len(next_action) != expected_count:
                issue("next_action_count_does_not_match_turn")
            if len(next_action) != len(set(next_action)):
                issue("next_action_contains_duplicates")
            expected_event = _next_event(full_timeline, turn)
            if expected_event is None or tuple(next_action) != tuple(
                normalize_name(value) for value in expected_event.monsters
            ):
                issue("next_action_does_not_match_full_events")

            occupied = {normalize_name(value) for value in my_picks + enemy_picks}
            if any(value in occupied for value in next_action):
                issue("next_action_already_selected")

            final_enemy = {normalize_name(value) for value in full_timeline.picks_for("enemy")}
            ban_slug = normalize_name(sample.get("ban_slug"))
            if ban_slug and ban_slug not in final_enemy:
                issue("ban_not_in_enemy_final_team")

        if sample_invalid:
            invalid_samples += 1

    return {
        "sample_count": len(samples),
        "issue_counts": dict(issue_counts),
        "issue_examples": examples,
        "valid_sample_count": len(samples) - invalid_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit canonical chronological BP samples.")
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
