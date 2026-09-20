"""Shared BP training data loading and action-label construction.

This module contains the normalization helpers that turn replay rows into
Shared BP Core pick and Ban examples.
"""

from __future__ import annotations

import json
import os
import sqlite3
from tqdm import tqdm

from bp_core.action_space import DraftTimeline, canonicalize_pair, validate_timeline
from bp_core.bp_common import (
    MonsterVocab,
    infer_pick_count,
    side_for_order,
)
from bp_core.data_common import safe_pickle_loads
from bp_core.names import PAD_TOKEN, normalize_name


FEATURE_FETCH_SIZE = 2048


def load_sqlite_samples(path, table_name="bp_samples", fetch_size=FEATURE_FETCH_SIZE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练样本数据库不存在: {path}")
    conn = sqlite3.connect(str(path))
    try:
        total = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        cursor = conn.execute(f"SELECT record FROM {table_name} ORDER BY key")
        samples = []
        with tqdm(total=total, desc=f"load {table_name}", unit="row") as progress:
            while True:
                rows = cursor.fetchmany(int(fetch_size))
                if not rows:
                    break
                samples.extend(safe_pickle_loads(row[0]) for row in rows)
                progress.update(len(rows))
        return samples
    finally:
        conn.close()


def load_samples(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".sqlite", ".db"}:
        samples = load_sqlite_samples(path, "bp_samples")
    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as file:
            samples = [json.loads(line) for line in file if line.strip()]
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            samples = json.load(file)
    else:
        raise ValueError(
            f"不支持的训练样本格式: {suffix or '<无扩展名>'}；"
            "请使用 SQLite、JSONL 或 JSON。"
        )
    if not isinstance(samples, list):
        raise ValueError("训练样本根对象必须是列表")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("训练样本列表只能包含对象")
    return samples


def extract_all_slugs_from_sample(sample):
    out = []
    for raw_event in sample["full_events"]:
        if not isinstance(raw_event, dict):
            raise ValueError("full_events 只能包含事件对象")
        for value in raw_event["monsters"]:
            slug = normalize_name(value)
            if not slug:
                raise ValueError("full_events 不能包含空魔灵")
            out.append(slug)
    for key in ("next_action_slugs",):
        values = sample[key]
        if isinstance(values, str):
            values = (values,)
        for value in values:
            slug = normalize_name(value)
            if not slug:
                raise ValueError(f"{key} 不能包含空魔灵")
            out.append(slug)
    for ban_key in ("ban_slug", "my_ban_slug", "enemy_ban_slug"):
        ban_slug = normalize_name(sample.get(ban_key))
        if ban_slug:
            out.append(ban_slug)
    for key in ("my_leader_slug", "enemy_leader_slug"):
        value = sample.get(key)
        if value:
            slug = normalize_name(value)
            if not slug:
                raise ValueError(f"{key} 不能包含空魔灵")
            out.append(slug)
    return out


def normalize_final_pick_slugs(sample, my_side=True):
    side = "me" if my_side else "enemy"
    timeline = DraftTimeline(sample["full_events"])
    values = [normalize_name(value) for value in timeline.picks_for(side)]
    if any(not value for value in values):
        raise ValueError("完整时间线包含空魔灵")
    return values


def parse_result_label(sample):
    value = normalize_name(sample.get("result"))
    if value in {"win", "victory", "winner"}:
        return 1.0
    if value in {"lose", "loss", "defeat"}:
        return 0.0
    return None


def perspective_events(events, own_side):
    timeline = DraftTimeline(events)
    return [
        {
            **event.to_dict(),
            "side": "me" if event.side == own_side else "enemy",
        }
        for event in timeline.events
    ]


def build_positive_actions(samples, vocab: MonsterVocab):
    """Build single and unordered-pair pick labels for the shared network."""
    positives = []
    for sample in samples:
        current_side = normalize_name(sample["current_side"])
        if current_side not in {"me", "enemy"}:
            raise ValueError("current_side must be me or enemy")
        timeline = DraftTimeline(sample["events"])
        full_timeline = DraftTimeline(sample["full_events"])
        global_first_pick_side = normalize_name(sample["first_pick_side"])
        if global_first_pick_side not in {"me", "enemy"}:
            raise ValueError("first_pick_side must be me or enemy")
        validate_timeline(timeline, global_first_pick_side, allow_bans=False)
        validate_timeline(full_timeline, global_first_pick_side, allow_bans=False, complete=True)
        if full_timeline.slots_used != 10 or timeline.slots_used != max(0, int(sample["turn"]) - 1):
            raise ValueError("sample timelines do not match the canonical turn")
        if any(event.action_type != "pick" for event in full_timeline.events):
            raise ValueError("full_events 只能包含 pick 事件")
        my_picks = list(timeline.picks_for("me"))
        enemy_picks = list(timeline.picks_for("enemy"))
        own_picks = my_picks if current_side == "me" else enemy_picks
        opponent_picks = enemy_picks if current_side == "me" else my_picks
        turn = int(sample["turn"])
        if current_side != side_for_order(turn, global_first_pick_side):
            raise ValueError("current_side does not match turn and first_pick_side")
        if any(
            prefix.canonical_key != full.canonical_key
            for prefix, full in zip(timeline.events, full_timeline.events)
        ):
            raise ValueError("events is not the canonical full_events prefix")
        first_pick_side = 1.0 if global_first_pick_side == current_side else 0.0
        relative_first_pick_side = "me" if first_pick_side > 0.5 else "enemy"
        pick_count = infer_pick_count(own_picks, opponent_picks, relative_first_pick_side)
        current_turn = [normalize_name(value) for value in sample["next_action_slugs"]]
        if any(not value for value in current_turn):
            raise ValueError("next_action_slugs 不能包含空魔灵")
        if len(current_turn) != pick_count:
            raise ValueError("next_action_slugs does not match the canonical BP event size")
        events = perspective_events(sample["events"], current_side)
        result_label = parse_result_label(sample)
        outcome = None if result_label is None else (
            result_label if current_side == "me" else 1.0 - result_label
        )
        if pick_count == 1:
            first_id = vocab.get_id(current_turn[0])
            if first_id <= 1:
                raise ValueError(f"next action is outside the Shared vocabulary: {current_turn[0]}")
            positives.append(
                {
                    "candidate": (first_id, vocab.get_id(PAD_TOKEN)),
                    "is_pair": False,
                    "turn": turn,
                    "first_pick_side": first_pick_side,
                    "outcome": outcome,
                    "events": events,
                }
            )
        elif pick_count == 2:
            first_id = vocab.get_id(current_turn[0])
            second_id = vocab.get_id(current_turn[1])
            if first_id <= 1 or second_id <= 1 or first_id == second_id:
                raise ValueError("pair action contains an unknown or duplicate monster")
            positives.append(
                {
                    "candidate": canonicalize_pair(first_id, second_id, key=int),
                    "is_pair": True,
                    "turn": turn,
                    "first_pick_side": first_pick_side,
                    "outcome": outcome,
                    "events": events,
                }
            )
        else:
            raise ValueError(f"unsupported canonical pick count: {pick_count}")

    return positives

def build_ban_positive_actions(samples, vocab: MonsterVocab):
    positives = []
    seen = set()
    for sample in samples:
        full_timeline = DraftTimeline(sample["full_events"])
        first_side = normalize_name(sample["first_pick_side"])
        validate_timeline(full_timeline, first_side, allow_bans=False, complete=True)
        if (
            full_timeline.slots_used != 10
            or any(event.action_type != "pick" for event in full_timeline.events)
        ):
            raise ValueError("ban sample requires a complete pick timeline")
        final_my = normalize_final_pick_slugs(sample, my_side=True)
        final_enemy = normalize_final_pick_slugs(sample, my_side=False)
        if len(final_my) != 5 or len(final_enemy) != 5:
            raise ValueError("full_events must contain five picks per side")
        if first_side not in {"me", "enemy"}:
            raise ValueError("first_pick_side must be me or enemy")
        enemy_set = set(final_enemy)
        ban_slug = normalize_name(sample["ban_slug"])
        if not ban_slug:
            continue
        if ban_slug not in enemy_set:
            raise ValueError("ban_slug must identify an enemy final pick")
        ban_id = vocab.get_id(ban_slug)
        enemy_ids = [vocab.get_id(x) for x in final_enemy if vocab.get_id(x) > 1]
        if (
            ban_id <= 1
            or ban_id >= len(vocab)
            or len(enemy_ids) != 5
            or len(set(enemy_ids)) != 5
            or ban_id not in enemy_ids
        ):
            raise ValueError("ban sample contains an unknown or duplicate monster")
        match_id = str(sample.get("match_id") or "")
        key = (match_id, tuple(final_my), tuple(final_enemy), ban_slug)
        if key in seen:
            continue
        seen.add(key)
        positives.append(
            {
                "candidate": ban_id,
                "enemy_candidate_ids": enemy_ids,
                "turn": 11,
                "first_pick_side": 1.0 if first_side == "me" else 0.0,
                "outcome": parse_result_label(sample),
                "events": list(sample["full_events"]),
            }
        )
    return positives
