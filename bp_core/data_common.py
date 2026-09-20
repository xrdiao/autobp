from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

from bp_core.action_space import validate_timeline
from bp_core.names import normalize_name, slugify_name


PICK_SIDE_PATTERN = ("me", "enemy", "enemy", "me", "me", "enemy", "enemy", "me", "me", "enemy")


class _SafeUnpickler(pickle.Unpickler):
    _ALLOWED_BUILTINS = {
        "bool", "bytes", "complex", "dict", "float", "frozenset", "int",
        "list", "set", "str", "tuple",
    }

    def find_class(self, module, name):
        if module == "builtins" and name in self._ALLOWED_BUILTINS:
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"unsupported pickle global: {module}.{name}")


def safe_pickle_loads(payload):
    """Load primitive records written by this project without importing classes."""
    return _SafeUnpickler(BytesIO(payload)).load()


def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, *, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))
    tmp_path.replace(path)


def open_record_db(path, table_name, indexed_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    column_sql = ",\n            ".join(f"{name} TEXT" for name in indexed_columns)
    if column_sql:
        column_sql += ","
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            key TEXT PRIMARY KEY,
            {column_sql}
            record BLOB NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    for name in indexed_columns:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{name} ON {table_name}({name})")
    return conn


def find_existing_record_keys(conn, table_name, keys, chunk_size=500):
    unique_keys = list(dict.fromkeys(str(key) for key in keys if key))
    if not unique_keys:
        return set()

    existing = set()
    for start in range(0, len(unique_keys), max(1, int(chunk_size))):
        chunk = unique_keys[start : start + max(1, int(chunk_size))]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT key FROM {table_name} WHERE key IN ({placeholders})",
            chunk,
        ).fetchall()
        existing.update(row[0] for row in rows)
    return existing


def upsert_record_rows(path, table_name, rows, indexed_columns, key_fn):
    rows = list(rows or [])
    if not rows:
        return 0, 0, 0
    conn = open_record_db(path, table_name, indexed_columns)
    skipped = 0
    placeholders = ", ".join(["?"] * (len(indexed_columns) + 3))
    columns_sql = ", ".join(["key", *indexed_columns, "record", "updated_at"])
    update_columns_sql = ", ".join(
        [f"{name}=excluded.{name}" for name in indexed_columns] + ["record=excluded.record", "updated_at=excluded.updated_at"]
    )
    now_value = int(time.time())
    try:
        prepared = []
        for row in rows:
            key = str(key_fn(row) or "")
            if not key:
                skipped += 1
                continue
            prepared.append(
                (
                    key,
                    [
                        key,
                        *[None if row.get(name) is None else str(row.get(name)) for name in indexed_columns],
                        sqlite3.Binary(pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)),
                        now_value,
                    ],
                )
            )

        if not prepared:
            return 0, 0, skipped

        existing_keys = find_existing_record_keys(conn, table_name, [key for key, _values in prepared])
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
                f"""
                INSERT INTO {table_name} ({columns_sql})
                VALUES ({placeholders})
                ON CONFLICT(key) DO UPDATE SET {update_columns_sql}
                """,
                values,
            )
    finally:
        conn.close()
    return appended, replaced, skipped


def load_record_rows(path, table_name, where_sql="", params=(), order_by="key"):
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(str(path))
    try:
        sql = f"SELECT record FROM {table_name}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        rows = conn.execute(sql, tuple(params or ())).fetchall()
        return [safe_pickle_loads(row[0]) for row in rows]
    finally:
        conn.close()


def count_record_rows(path, table_name, where_sql="", params=()):
    if not os.path.exists(path):
        return 0
    conn = sqlite3.connect(str(path))
    try:
        sql = f"SELECT COUNT(*) FROM {table_name}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        return int(conn.execute(sql, tuple(params or ())).fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


class NameResolver:
    def __init__(self, alias_map=None, display_name_map=None):
        alias_map = alias_map or {}
        display_name_map = display_name_map or {}
        self.alias_map = {
            normalize_name(key): normalize_name(value)
            for key, value in alias_map.items()
            if key and value
        }
        self.display_name_map = {
            normalize_name(key): str(value).strip()
            for key, value in display_name_map.items()
            if key and value
        }
        self._slug_cache = {}
        for slug, display_name in self.display_name_map.items():
            self.alias_map.setdefault(slug, slug)
            self.alias_map.setdefault(normalize_name(display_name), slug)
            self.alias_map.setdefault(slugify_name(display_name), slug)

    def to_slug(self, value):
        cache_key = "" if value is None else str(value)
        if cache_key in self._slug_cache:
            return self._slug_cache[cache_key]

        normalized = normalize_name(value)
        if not normalized:
            self._slug_cache[cache_key] = ""
            return ""
        generated = slugify_name(value)
        sanitized = generated.replace("[", "").replace("]", "")
        for key in (normalized, generated, sanitized):
            resolved = self.alias_map.get(key)
            if resolved:
                result = normalize_name(resolved)
                self._slug_cache[cache_key] = result
                return result
        self._slug_cache[cache_key] = ""
        return ""

    def normalize_slug_list(self, values):
        return [slug for slug in (self.to_slug(value) for value in (values or [])) if slug]

    def to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug, slug)


def _record_name_slug_pair(alias_counters, display_counters, name, slug):
    name_text = str(name).strip() if name is not None else ""
    slug_text = normalize_name(slug)
    if not name_text or not slug_text:
        return
    for key in {normalize_name(name_text), slugify_name(name_text), slug_text}:
        if key:
            alias_counters.setdefault(key, Counter())[slug_text] += 1
    if normalize_name(name_text) != slug_text:
        display_counters.setdefault(slug_text, Counter())[name_text] += 1


def build_name_resolution_resources(samples):
    alias_counters = {}
    display_counters = {}

    for sample in samples or []:
        # ``full_events`` is the authoritative name source.  ``events`` is
        # always a prefix of it, so scanning it again only multiplies the
        # resolver work without adding aliases.
        full_events = sample.get("full_events")
        if not isinstance(full_events, (list, tuple)):
            raise ValueError("样本缺少 canonical full_events")
        for raw_event in full_events:
            if not isinstance(raw_event, dict):
                raise ValueError("full_events 中存在非法事件")
            slugs = raw_event.get("monsters")
            if not isinstance(slugs, (list, tuple)):
                raise ValueError("full_events 事件缺少 monsters")
            names = raw_event.get("monster_names") or ()
            for index, slug in enumerate(slugs):
                name = names[index] if index < len(names) else slug
                _record_name_slug_pair(alias_counters, display_counters, name, slug)
        next_action = sample.get("next_action_slugs")
        if not isinstance(next_action, (list, tuple)):
            raise ValueError("样本缺少 canonical next_action_slugs")
        for slug in next_action:
            _record_name_slug_pair(alias_counters, display_counters, slug, slug)
        for ban_name_key, ban_slug_key in (
            ("ban_name", "ban_slug"),
            ("my_ban", "my_ban_slug"),
            ("enemy_ban", "enemy_ban_slug"),
        ):
            _record_name_slug_pair(
                alias_counters,
                display_counters,
                sample.get(ban_name_key),
                sample.get(ban_slug_key),
            )
        for leader_key in ("my_leader", "enemy_leader"):
            slug_key = f"{leader_key}_slug"
            _record_name_slug_pair(
                alias_counters,
                display_counters,
                sample.get(leader_key),
                sample.get(slug_key),
            )

    name_slug_aliases = {key: counter.most_common(1)[0][0] for key, counter in alias_counters.items() if counter}
    display_name_map = {slug: counter.most_common(1)[0][0] for slug, counter in display_counters.items() if counter}
    return {
        "name_slug_aliases": name_slug_aliases,
        "display_name_map": display_name_map,
        "image_name_map": dict(display_name_map),
    }


def normalize_bp_sample(sample, resolver):
    item = dict(sample)

    required = ("events", "full_events", "next_action_slugs", "turn", "first_pick_side", "current_side")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"样本缺少 canonical 字段: {', '.join(missing)}")

    def normalize_events(raw_events, field_name, *, require_nonempty):
        if not isinstance(raw_events, (list, tuple)):
            raise ValueError(f"样本缺少 canonical {field_name} 事件列表")
        if require_nonempty and not raw_events:
            raise ValueError(f"样本的 {field_name} 不能为空")
        normalized = []
        for index, raw_event in enumerate(raw_events, start=1):
            if not isinstance(raw_event, dict):
                raise ValueError(f"{field_name}[{index}] 不是事件对象")
            event = dict(raw_event)
            raw_values = event.get("monsters")
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(f"{field_name}[{index}] 缺少 monsters")
            values = resolver.normalize_slug_list(raw_values)
            if len(values) != len(raw_values) or len(values) not in {1, 2}:
                raise ValueError(f"{field_name}[{index}] 包含未解析或非法数量的魔灵")
            event["monsters"] = values
            if normalize_name(event.get("action_type") or "pick") != "pick":
                raise ValueError(f"{field_name}[{index}] 不是 pick 事件")
            event["action_type"] = "pick"
            event["side"] = normalize_name(event.get("side"))
            if event["side"] not in {"me", "enemy"}:
                raise ValueError(f"{field_name}[{index}] side 无效")
            event["order"] = index
            event["is_pair"] = len(values) == 2
            names = event.get("monster_names") or []
            event["monster_names"] = [
                str(names[name_index]).strip()
                if name_index < len(names) and names[name_index]
                else value
                for name_index, value in enumerate(values)
            ]
            normalized.append(event)
        return normalized

    item["events"] = normalize_events(item["events"], "events", require_nonempty=False)
    item["full_events"] = normalize_events(item["full_events"], "full_events", require_nonempty=True)

    first_pick_side = normalize_name(item["first_pick_side"])
    current_side = normalize_name(item["current_side"])
    if first_pick_side not in {"me", "enemy"}:
        raise ValueError("first_pick_side 必须是 me 或 enemy")
    if current_side not in {"me", "enemy"}:
        raise ValueError("current_side 必须是 me 或 enemy")
    try:
        turn = int(item["turn"])
    except (TypeError, ValueError) as exc:
        raise ValueError("turn 必须是整数") from exc
    if turn < 1 or turn > 10:
        raise ValueError("训练样本 turn 必须位于 1 到 10")

    prefix_timeline = _timeline_from_events(item["events"])
    full_timeline = _timeline_from_events(item["full_events"])
    validate_timeline(prefix_timeline, first_pick_side, allow_bans=False)
    validate_timeline(full_timeline, first_pick_side, allow_bans=False, complete=True)
    if full_timeline.slots_used != 10:
        raise ValueError("full_events 必须完整包含十个 pick 槽位")
    full_values = [normalize_name(value) for value in full_timeline.picks_for("me") + full_timeline.picks_for("enemy")]
    if len(full_values) != 10 or not all(full_values) or len(set(full_values)) != 10:
        raise ValueError("full_events 必须包含十个互不重复的魔灵")
    slot = 1
    for event in full_timeline.events:
        for _monster in event.monsters:
            if event.side != _side_for_pick_slot(slot, first_pick_side):
                raise ValueError("full_events 的 side 顺序与 BP 轮次不一致")
            slot += 1
    if len(full_timeline.picks_for("me")) != 5 or len(full_timeline.picks_for("enemy")) != 5:
        raise ValueError("full_events 必须让双方各有五个 pick")
    if prefix_timeline.slots_used != turn - 1:
        raise ValueError("events 前缀与 turn 不一致")
    if len(prefix_timeline.events) >= len(full_timeline.events) and turn <= 10:
        raise ValueError("events 必须在下一动作前截断")
    for prefix_event, full_event in zip(prefix_timeline.events, full_timeline.events):
        if prefix_event.canonical_key != full_event.canonical_key:
            raise ValueError("events 不是 full_events 的严格前缀")
    expected_side = _side_for_pick_slot(turn, first_pick_side)
    if current_side != expected_side:
        raise ValueError("current_side 与 turn/first_pick_side 不一致")

    raw_next = item.get("next_action_slugs")
    if not isinstance(raw_next, (list, tuple)) or len(raw_next) not in {1, 2}:
        raise ValueError("样本缺少合法的 next_action_slugs")
    item["next_action_slugs"] = resolver.normalize_slug_list(raw_next)
    if len(item["next_action_slugs"]) != len(raw_next):
        raise ValueError("next_action_slugs 包含未解析的魔灵")
    if len(item["next_action_slugs"]) == 2:
        item["next_action_slugs"] = sorted(item["next_action_slugs"])
    next_event = _next_event_from_timeline(full_timeline, turn)
    if next_event is None:
        raise ValueError("full_events 中找不到 turn 对应的下一事件")
    expected_action = tuple(normalize_name(value) for value in next_event.monsters)
    if tuple(item["next_action_slugs"]) != expected_action:
        raise ValueError("next_action_slugs 与 full_events 的下一事件不一致")
    raw_my_ban = item.get("my_ban_slug") or item.get("ban_slug")
    raw_enemy_ban = item.get("enemy_ban_slug") or item.get("enemy_ban")
    item["my_ban_slug"] = resolver.to_slug(raw_my_ban) if raw_my_ban else ""
    item["enemy_ban_slug"] = resolver.to_slug(raw_enemy_ban) if raw_enemy_ban else ""
    if raw_my_ban and not item["my_ban_slug"]:
        raise ValueError("my_ban_slug 无法解析")
    if raw_enemy_ban and not item["enemy_ban_slug"]:
        raise ValueError("enemy_ban_slug 无法解析")
    enemy_final = set(normalize_name(value) for value in full_timeline.picks_for("enemy"))
    my_final = set(normalize_name(value) for value in full_timeline.picks_for("me"))
    if item["my_ban_slug"] and item["my_ban_slug"] not in enemy_final:
        raise ValueError("my_ban_slug 必须属于敌方最终阵容")
    if item["enemy_ban_slug"] and item["enemy_ban_slug"] not in my_final:
        raise ValueError("enemy_ban_slug 必须属于我方最终阵容")
    # Keep the historical field as the current player's Ban alias.
    item["ban_slug"] = item["my_ban_slug"]
    for leader_key in ("my_leader_slug", "enemy_leader_slug"):
        raw_leader = item.get(leader_key)
        item[leader_key] = resolver.to_slug(raw_leader) if raw_leader else ""
        if raw_leader and not item[leader_key]:
            raise ValueError(f"{leader_key} 无法解析")
    if item["my_leader_slug"] and item["my_leader_slug"] not in set(full_timeline.picks_for("me")):
        raise ValueError("my_leader_slug 必须属于我方最终阵容")
    if item["enemy_leader_slug"] and item["enemy_leader_slug"] not in enemy_final:
        raise ValueError("enemy_leader_slug 必须属于敌方最终阵容")
    return item


def normalize_relation_sample(sample, resolver):
    """Expose legacy trainer fields from the canonical event sample.

    The relation and prefix trainers are retained for their stronger learned
    models, while the repository keeps one event-based sample schema.  This
    bridge derives the old side lists without changing event order or pair
    semantics.
    """
    item = normalize_bp_sample(sample, resolver)

    def side_picks(events, side):
        values = []
        for event in events:
            if event.get("side") == side:
                values.extend(event.get("monsters") or [])
        return values

    prefix_events = item["events"]
    full_events = item["full_events"]
    my_prefix = side_picks(prefix_events, "me")
    enemy_prefix = side_picks(prefix_events, "enemy")
    my_final = side_picks(full_events, "me")
    enemy_final = side_picks(full_events, "enemy")
    item["my_picks_slugs"] = list(my_prefix)
    item["enemy_picks_slugs"] = list(enemy_prefix)
    item["final_my_pick_slugs"] = list(my_final)
    item["final_enemy_pick_slugs"] = list(enemy_final)
    item["remaining_my_pick_slugs"] = list(my_final[len(my_prefix) :])
    item["remaining_enemy_pick_slugs"] = list(enemy_final[len(enemy_prefix) :])
    next_action = list(item.get("next_action_slugs") or [])
    item["next_pick_slug"] = next_action[0] if next_action else ""
    item["my_ban_slug"] = item.get("my_ban_slug") or ""
    item["enemy_ban_slug"] = item.get("enemy_ban_slug") or ""
    item["my_leader_slug"] = item.get("my_leader_slug") or ""
    item["enemy_leader_slug"] = item.get("enemy_leader_slug") or ""
    return item


def _timeline_from_events(events):
    """Build a strict pick timeline after names have been resolved."""
    from bp_core.action_space import DraftTimeline

    return DraftTimeline(events)


def _side_for_pick_slot(slot, first_pick_side):
    if first_pick_side == "me":
        return PICK_SIDE_PATTERN[int(slot) - 1]
    return tuple("enemy" if side == "me" else "me" for side in PICK_SIDE_PATTERN)[int(slot) - 1]


def _next_event_from_timeline(timeline, turn):
    slots = 0
    for event in timeline.events:
        if slots + 1 == int(turn):
            return event
        slots += event.count
    return None
