from __future__ import annotations

import json
import os
import pickle
import sqlite3
from collections import Counter
from pathlib import Path

from bp_transformer_common import normalize_name, slugify_fallback


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


def load_pt(path, default=None):
    if not os.path.exists(path):
        return [] if default is None else default
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_pt(data, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp_path)
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


def upsert_record_rows(path, table_name, rows, indexed_columns, key_fn):
    rows = list(rows or [])
    if not rows:
        return 0, 0, 0
    conn = open_record_db(path, table_name, indexed_columns)
    appended = 0
    replaced = 0
    skipped = 0
    placeholders = ", ".join(["?"] * (len(indexed_columns) + 3))
    columns_sql = ", ".join(["key", *indexed_columns, "record", "updated_at"])
    update_columns_sql = ", ".join(
        [f"{name}=excluded.{name}" for name in indexed_columns] + ["record=excluded.record", "updated_at=excluded.updated_at"]
    )
    now_value = int(__import__("time").time())
    try:
        with conn:
            for row in rows:
                key = str(key_fn(row) or "")
                if not key:
                    skipped += 1
                    continue
                existed = conn.execute(f"SELECT 1 FROM {table_name} WHERE key = ?", (key,)).fetchone() is not None
                values = [
                    key,
                    *[None if row.get(name) is None else str(row.get(name)) for name in indexed_columns],
                    sqlite3.Binary(pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)),
                    now_value,
                ]
                conn.execute(
                    f"""
                    INSERT INTO {table_name} ({columns_sql})
                    VALUES ({placeholders})
                    ON CONFLICT(key) DO UPDATE SET {update_columns_sql}
                    """,
                    values,
                )
                if existed:
                    replaced += 1
                else:
                    appended += 1
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
        return [pickle.loads(row[0]) for row in rows]
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
        for slug, display_name in self.display_name_map.items():
            self.alias_map.setdefault(slug, slug)
            self.alias_map.setdefault(normalize_name(display_name), slug)
            self.alias_map.setdefault(slugify_fallback(display_name), slug)

    def to_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        fallback = slugify_fallback(value)
        sanitized_fallback = fallback.replace("[", "").replace("]", "")
        for key in (normalized, fallback, sanitized_fallback):
            resolved = self.alias_map.get(key)
            if resolved:
                return normalize_name(resolved)
        return sanitized_fallback or fallback or normalized

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
    for key in {normalize_name(name_text), slugify_fallback(name_text), slug_text}:
        if key:
            alias_counters.setdefault(key, Counter())[slug_text] += 1
    display_counters.setdefault(slug_text, Counter())[name_text] += 1


def build_name_resolution_resources(samples):
    alias_counters = {}
    display_counters = {}

    def record_paired_lists(names, slugs):
        for name, slug in zip(names or [], slugs or []):
            _record_name_slug_pair(alias_counters, display_counters, name, slug)

    for sample in samples or []:
        _record_name_slug_pair(alias_counters, display_counters, sample.get("next_pick"), sample.get("next_pick_slug"))
        _record_name_slug_pair(alias_counters, display_counters, sample.get("my_ban"), sample.get("my_ban_slug"))
        _record_name_slug_pair(alias_counters, display_counters, sample.get("enemy_ban"), sample.get("enemy_ban_slug"))
        _record_name_slug_pair(alias_counters, display_counters, sample.get("my_leader"), sample.get("my_leader_slug"))
        _record_name_slug_pair(alias_counters, display_counters, sample.get("enemy_leader"), sample.get("enemy_leader_slug"))
        record_paired_lists(sample.get("my_picks"), sample.get("my_picks_slugs"))
        record_paired_lists(sample.get("enemy_picks"), sample.get("enemy_picks_slugs"))
        record_paired_lists(sample.get("remaining_my_picks"), sample.get("remaining_my_pick_slugs"))
        record_paired_lists(sample.get("remaining_enemy_picks"), sample.get("remaining_enemy_pick_slugs"))
        record_paired_lists(sample.get("final_my_picks"), sample.get("final_my_pick_slugs"))
        record_paired_lists(sample.get("final_enemy_picks"), sample.get("final_enemy_pick_slugs"))

    name_slug_aliases = {key: counter.most_common(1)[0][0] for key, counter in alias_counters.items() if counter}
    display_name_map = {slug: counter.most_common(1)[0][0] for slug, counter in display_counters.items() if counter}
    return {
        "name_slug_aliases": name_slug_aliases,
        "display_name_map": display_name_map,
        "image_name_map": dict(display_name_map),
    }


def normalize_relation_sample(sample, resolver):
    item = dict(sample)
    item["my_picks_slugs"] = list(item.get("my_picks_slugs") or resolver.normalize_slug_list(item.get("my_picks")))
    item["enemy_picks_slugs"] = list(item.get("enemy_picks_slugs") or resolver.normalize_slug_list(item.get("enemy_picks")))
    item["next_pick_slug"] = item.get("next_pick_slug") or resolver.to_slug(item.get("next_pick"))
    item["my_ban_slug"] = item.get("my_ban_slug") or resolver.to_slug(item.get("my_ban"))
    item["enemy_ban_slug"] = item.get("enemy_ban_slug") or resolver.to_slug(item.get("enemy_ban"))
    item["my_leader_slug"] = item.get("my_leader_slug") or resolver.to_slug(item.get("my_leader"))
    item["enemy_leader_slug"] = item.get("enemy_leader_slug") or resolver.to_slug(item.get("enemy_leader"))
    item["remaining_my_pick_slugs"] = list(
        item.get("remaining_my_pick_slugs") or resolver.normalize_slug_list(item.get("remaining_my_picks"))
    )
    item["remaining_enemy_pick_slugs"] = list(
        item.get("remaining_enemy_pick_slugs") or resolver.normalize_slug_list(item.get("remaining_enemy_picks"))
    )
    item["final_my_pick_slugs"] = list(item.get("final_my_pick_slugs") or resolver.normalize_slug_list(item.get("final_my_picks")))
    item["final_enemy_pick_slugs"] = list(
        item.get("final_enemy_pick_slugs") or resolver.normalize_slug_list(item.get("final_enemy_picks"))
    )
    return item
