import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from itertools import combinations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from bp_core.data_common import NameResolver, build_name_resolution_resources, normalize_relation_sample
from bp_core.relation_common import (
    BPDraftRelationNetV2,
    MonsterVocab,
    infer_pick_count,
    normalize_slug_list,
    resolve_monster_slug,
)
from bp_core.transformer_common import PAD_TOKEN, PROJECT_ROOT, UNK_TOKEN, normalize_name, slugify_fallback

DATA_PATH = str(PROJECT_ROOT / "data" / "bp_samples.sqlite")
SAVE_DIR = str(PROJECT_ROOT / "models")
MODEL_PATH = os.path.join(SAVE_DIR, "bp_relation_net_v2.pt")
META_PATH = os.path.join(SAVE_DIR, "bp_relation_meta_v2.json")
VOCAB_PATH = os.path.join(SAVE_DIR, "bp_relation_vocab_v2.json")
SPLIT_MANIFEST_PATH = os.path.join(SAVE_DIR, "bp_relation_split_manifest.json")
PREPROCESSED_CACHE_PATH = os.path.join(SAVE_DIR, "bp_relation_preprocessed_cache.pt")
FEATURE_CACHE_PATH = os.path.join(SAVE_DIR, "bp_relation_feature_cache.sqlite")

BATCH_SIZE = 1024
EPOCHS = 10
FINETUNE_EPOCHS = 3
LR = 1e-3
FINETUNE_LR = 3e-4
WEIGHT_DECAY = 1e-5
EMBED_DIM = 256
STATE_DIM = 256
ACTION_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.15
NEGATIVES_PER_POSITIVE = 4
BAN_LOSS_WEIGHT = 0.8
VAL_RATIO = 0.1
SEED = 42
SPLIT_MANIFEST_VERSION = 1
PREPROCESS_VERSION = 6
FEATURE_CACHE_SCHEMA_VERSION = 3
FEATURE_FETCH_SIZE = 2048
FEATURE_CACHE_COMMIT_GROUPS = 512
FEATURE_CACHE_WORKERS = max(1, min(4, (os.cpu_count() or 1)))
RELATION_ARCHITECTURE_VERSION = 3
HIGH_FREQ_NEGATIVE_RATIO = 0.20
NEAR_FREQ_NEGATIVE_RATIO = 0.20
OUTCOME_LOSS_WEIGHT = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune the BP relation recommender.")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--meta-path", default=META_PATH)
    parser.add_argument("--vocab-path", default=VOCAB_PATH)
    parser.add_argument("--split-manifest-path", default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--preprocessed-cache-path", default=PREPROCESSED_CACHE_PATH)
    parser.add_argument("--feature-cache-path", default=FEATURE_CACHE_PATH)
    parser.add_argument("--feature-cache-commit-groups", type=int, default=FEATURE_CACHE_COMMIT_GROUPS)
    parser.add_argument("--feature-cache-workers", type=int, default=FEATURE_CACHE_WORKERS)
    parser.add_argument("--rebuild-preprocessed-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--negatives-per-positive", type=int, default=NEGATIVES_PER_POSITIVE)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.25)
    parser.add_argument("--hard-negative-candidate-pool", type=int, default=64)
    parser.add_argument("--pick-eval-candidate-cap", type=int, default=96)
    parser.add_argument("--pick-eval-max-samples", type=int, default=2048)
    parser.add_argument("--ban-loss-weight", type=float, default=BAN_LOSS_WEIGHT)
    parser.add_argument("--outcome-loss-weight", type=float, default=OUTCOME_LOSS_WEIGHT)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--new-data-weight", type=float, default=4.0)
    parser.add_argument("--old-data-replay-ratio", type=float, default=0.35)
    parser.add_argument("--fine-tune-new-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true", default=True)
    parser.add_argument("--from-scratch", dest="fine_tune", action="store_false")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(pin_memory=True, persistent_workers=True, amp=True)
    return parser.parse_args()


def log_progress(message):
    print(message, flush=True)


def load_sqlite_samples(path, table_name="bp_samples", fetch_size=FEATURE_FETCH_SIZE):
    if not os.path.exists(path):
        return []
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
                samples.extend(pickle.loads(row[0]) for row in rows)
                progress.update(len(rows))
        return samples
    finally:
        conn.close()


def load_samples(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".sqlite", ".db"}:
        return load_sqlite_samples(path, "bp_samples")
    if suffix == ".pt":
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    if suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def preprocessed_cache_signature(args, can_resume):
    return {
        "cache_version": 2,
        "preprocess_version": PREPROCESS_VERSION,
        "feature_cache": feature_cache_signature(args.feature_cache_path),
        "can_resume": bool(can_resume),
        "fine_tune": bool(args.fine_tune),
        "fine_tune_new_only": bool(args.fine_tune_new_only),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "rebuild_split": bool(args.rebuild_split),
    }


def load_preprocessed_cache(args, can_resume):
    if args.rebuild_preprocessed_cache or args.rebuild_split:
        return None
    if not args.preprocessed_cache_path or not os.path.exists(args.preprocessed_cache_path):
        return None
    try:
        cache = _torch_load_cache(args.preprocessed_cache_path)
    except Exception as exc:
        print(f"[WARN] failed to load preprocessed cache: {exc}")
        return None
    expected_signature = preprocessed_cache_signature(args, can_resume)
    if cache.get("signature") != expected_signature:
        return None
    return cache.get("payload")


def save_preprocessed_cache(args, can_resume, payload):
    if not args.preprocessed_cache_path:
        return
    cache_dir = os.path.dirname(args.preprocessed_cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    cache = {
        "signature": preprocessed_cache_signature(args, can_resume),
        "payload": payload,
    }
    torch.save(cache, args.preprocessed_cache_path)
    log_progress(f"Saved preprocessed cache: {args.preprocessed_cache_path}")


def _json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _group_samples_hash(group_samples):
    raw = "\n".join(sorted(stable_sample_hash(sample) for sample in group_samples))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _group_manifest_hash(group_entry):
    raw = "\n".join(group_entry.get("sample_hashes") or [])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def open_feature_cache(path):
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_groups (
            group_key TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            sample_hash TEXT NOT NULL,
            is_new_group INTEGER NOT NULL,
            preprocess_version INTEGER NOT NULL,
            vocab_hash TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_features (
            feature_id TEXT PRIMARY KEY,
            group_key TEXT NOT NULL,
            split TEXT NOT NULL,
            feature_type TEXT NOT NULL,
            is_new_group INTEGER NOT NULL,
            preprocess_version INTEGER NOT NULL,
            vocab_hash TEXT NOT NULL,
            record BLOB NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_group_features (
            group_key TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            is_new_group INTEGER NOT NULL,
            preprocess_version INTEGER NOT NULL,
            vocab_hash TEXT NOT NULL,
            pick_rows BLOB NOT NULL,
            ban_rows BLOB NOT NULL,
            monster_freq BLOB NOT NULL,
            pair_freq BLOB NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_features_group ON relation_features(group_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_features_type_split ON relation_features(feature_type, split)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_group_features_split ON relation_group_features(split)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_groups_split ON relation_groups(split)")
    return conn


def reset_feature_cache_files(path):
    for suffix in ("", "-wal", "-shm"):
        target = f"{path}{suffix}"
        if os.path.exists(target):
            os.remove(target)


def feature_cache_signature(path):
    if not path or not os.path.exists(path):
        return None
    conn = sqlite3.connect(str(path))
    try:
        group_row = conn.execute(
            """
            SELECT COUNT(*), COALESCE(MAX(updated_at), 0)
            FROM relation_groups
            WHERE preprocess_version = ?
            """,
            (PREPROCESS_VERSION,),
        ).fetchone()
        feature_rows = conn.execute(
            """
            SELECT split, COUNT(*), COALESCE(MAX(updated_at), 0)
            FROM relation_group_features
            WHERE preprocess_version = ?
            GROUP BY split
            ORDER BY split
            """,
            (PREPROCESS_VERSION,),
        ).fetchall()
        meta_rows = conn.execute("SELECT key, value FROM relation_meta ORDER BY key").fetchall()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return {
        "path": os.path.abspath(path),
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "group_count": int(group_row[0] or 0),
        "group_updated_at": int(group_row[1] or 0),
        "group_feature_counts": [(str(s), int(c), int(m or 0)) for s, c, m in feature_rows],
        "meta": {str(key): str(value) for key, value in meta_rows},
    }


def build_grouped_samples(samples):
    grouped_samples = {}
    for sample in tqdm(samples, desc="group samples", unit="sample"):
        group_key = sample.get("_split_group_key") or get_split_group_key(sample)
        sample["_split_group_key"] = group_key
        grouped_samples.setdefault(group_key, []).append(sample)
    return grouped_samples


def _load_cached_group_rows(conn):
    rows = conn.execute(
        """
        SELECT group_key, split, sample_count, sample_hash, is_new_group
        FROM relation_groups
        WHERE preprocess_version = ?
        """,
        (PREPROCESS_VERSION,),
    ).fetchall()
    return {
        row[0]: {
            "split": row[1],
            "sample_count": int(row[2]),
            "sample_hash": row[3],
            "is_new_group": bool(row[4]),
        }
        for row in rows
    }


def build_feature_cache_change_plan(grouped_samples, split_manifest, args, can_resume):
    if not args.feature_cache_path:
        return {
            "changed": True,
            "changed_group_keys": set(grouped_samples),
            "stale_group_keys": set(),
            "forced_rebuild": True,
            "cached_groups": {},
        }

    conn = open_feature_cache(args.feature_cache_path)
    try:
        cached_groups = _load_cached_group_rows(conn)
    finally:
        conn.close()

    force_rebuild = bool(args.rebuild_preprocessed_cache or args.rebuild_split or not can_resume)
    current_keys = set(grouped_samples)
    cached_keys = set(cached_groups)
    stale_keys = cached_keys - current_keys
    changed_keys = set(current_keys) if force_rebuild else set()

    if not force_rebuild:
        for group_key, group_samples in grouped_samples.items():
            group_entry = split_manifest["groups"][group_key]
            split = group_entry["split"]
            is_new_group = bool(group_samples[0].get("_is_new_group", False))
            sample_hash = _group_manifest_hash(group_entry)
            cached = cached_groups.get(group_key)
            if (
                cached is None
                or cached["split"] != split
                or cached["sample_count"] != len(group_samples)
                or cached["sample_hash"] != sample_hash
                or cached["is_new_group"] != is_new_group
            ):
                changed_keys.add(group_key)

    return {
        "changed": bool(force_rebuild or stale_keys or changed_keys),
        "changed_group_keys": changed_keys,
        "stale_group_keys": stale_keys,
        "forced_rebuild": force_rebuild,
        "cached_groups": cached_groups,
    }


FEATURE_INSERT_SQL = """
    INSERT INTO relation_features(
        feature_id, group_key, split, feature_type, is_new_group,
        preprocess_version, vocab_hash, record, updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


GROUP_FEATURE_UPSERT_SQL = """
    INSERT INTO relation_group_features(
        group_key, split, is_new_group, preprocess_version, vocab_hash,
        pick_rows, ban_rows, monster_freq, pair_freq, updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(group_key) DO UPDATE SET
        split = excluded.split,
        is_new_group = excluded.is_new_group,
        preprocess_version = excluded.preprocess_version,
        vocab_hash = excluded.vocab_hash,
        pick_rows = excluded.pick_rows,
        ban_rows = excluded.ban_rows,
        monster_freq = excluded.monster_freq,
        pair_freq = excluded.pair_freq,
        updated_at = excluded.updated_at
"""


GROUP_UPSERT_SQL = """
    INSERT INTO relation_groups(
        group_key, split, sample_count, sample_hash, is_new_group,
        preprocess_version, vocab_hash, updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(group_key) DO UPDATE SET
        split = excluded.split,
        sample_count = excluded.sample_count,
        sample_hash = excluded.sample_hash,
        is_new_group = excluded.is_new_group,
        preprocess_version = excluded.preprocess_version,
        vocab_hash = excluded.vocab_hash,
        updated_at = excluded.updated_at
"""


def commit_feature_cache_batch(conn, delete_group_keys, group_feature_values, group_values):
    if not delete_group_keys and not group_feature_values and not group_values:
        return
    with conn:
        if delete_group_keys:
            key_rows = [(key,) for key in delete_group_keys]
            conn.executemany("DELETE FROM relation_features WHERE group_key = ?", key_rows)
            conn.executemany("DELETE FROM relation_group_features WHERE group_key = ?", key_rows)
        if group_feature_values:
            conn.executemany(GROUP_FEATURE_UPSERT_SQL, group_feature_values)
        if group_values:
            conn.executemany(GROUP_UPSERT_SQL, group_values)


_FEATURE_WORKER_VOCAB = None
_FEATURE_WORKER_ALIAS_MAP = None


def init_feature_worker(vocab_dict, alias_map):
    global _FEATURE_WORKER_VOCAB, _FEATURE_WORKER_ALIAS_MAP
    _FEATURE_WORKER_VOCAB = MonsterVocab(vocab_dict)
    _FEATURE_WORKER_ALIAS_MAP = alias_map


def build_group_feature_result(task):
    group_key, group_samples, split, is_new_group, sample_hash = task
    vocab = _FEATURE_WORKER_VOCAB
    alias_map = _FEATURE_WORKER_ALIAS_MAP
    pick_rows = build_positive_actions(group_samples, vocab, alias_map)
    ban_rows = build_ban_positive_actions(group_samples, vocab, alias_map)
    monster_freq = Counter()
    pair_freq = Counter()
    if split == "train":
        monster_freq, pair_freq = collect_frequency_stats(group_samples, vocab, alias_map, show_progress=False)
    return {
        "group_key": group_key,
        "split": split,
        "is_new_group": bool(is_new_group),
        "sample_hash": sample_hash,
        "sample_count": len(group_samples),
        "pick_rows": pick_rows,
        "ban_rows": ban_rows,
        "monster_freq": dict(monster_freq),
        "pair_freq": dict(pair_freq),
    }


def build_group_feature_tasks(grouped_samples, split_manifest, group_keys):
    for group_key in group_keys:
        group_samples = grouped_samples[group_key]
        group_entry = split_manifest["groups"][group_key]
        yield (
            group_key,
            group_samples,
            group_entry["split"],
            bool(group_samples[0].get("_is_new_group", False)),
            _group_manifest_hash(group_entry),
        )


def iter_group_feature_results(grouped_samples, split_manifest, changed_group_keys, vocab, alias_map, worker_count):
    tasks = list(build_group_feature_tasks(grouped_samples, split_manifest, changed_group_keys))
    if worker_count <= 1 or len(tasks) <= 1:
        init_feature_worker(vocab.to_dict(), alias_map)
        for task in tasks:
            yield build_group_feature_result(task)
        return

    chunksize = max(1, min(128, len(tasks) // (worker_count * 8) or 1))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_feature_worker,
        initargs=(vocab.to_dict(), alias_map),
    ) as executor:
        yield from executor.map(build_group_feature_result, tasks, chunksize=chunksize)


def sync_relation_feature_cache(grouped_samples, split_manifest, vocab, alias_map, args, can_resume, change_plan=None):
    if not args.feature_cache_path:
        return {"changed": True, "updated_groups": 0, "deleted_groups": 0, "forced_rebuild": True}

    if change_plan is None:
        change_plan = build_feature_cache_change_plan(grouped_samples, split_manifest, args, can_resume)

    vocab_hash = _json_hash(vocab.to_dict())
    force_rebuild = bool(change_plan["forced_rebuild"])
    stale_keys = sorted(change_plan["stale_group_keys"])
    changed_group_keys = sorted(change_plan["changed_group_keys"])
    reset_existing_cache = force_rebuild or (
        not change_plan.get("cached_groups")
        and len(changed_group_keys) == len(grouped_samples)
        and os.path.exists(args.feature_cache_path)
    )
    if reset_existing_cache:
        reset_feature_cache_files(args.feature_cache_path)
        stale_keys = []

    conn = open_feature_cache(args.feature_cache_path)
    now_value = int(time.time())
    updated_groups = 0
    deleted_groups = 0
    changed = bool(change_plan["changed"])

    with conn:
        conn.execute("INSERT OR REPLACE INTO relation_meta(key, value) VALUES(?, ?)", ("schema_version", str(FEATURE_CACHE_SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO relation_meta(key, value) VALUES(?, ?)", ("preprocess_version", str(PREPROCESS_VERSION)))
        conn.execute("INSERT OR REPLACE INTO relation_meta(key, value) VALUES(?, ?)", ("vocab_hash", vocab_hash))
        if force_rebuild and not reset_existing_cache:
            conn.execute("DELETE FROM relation_features")
            conn.execute("DELETE FROM relation_group_features")
            conn.execute("DELETE FROM relation_groups")
            stale_keys = []
        elif stale_keys:
            for group_key in stale_keys:
                conn.execute("DELETE FROM relation_features WHERE group_key = ?", (group_key,))
                conn.execute("DELETE FROM relation_group_features WHERE group_key = ?", (group_key,))
                conn.execute("DELETE FROM relation_groups WHERE group_key = ?", (group_key,))
                deleted_groups += 1

    commit_groups = max(1, int(getattr(args, "feature_cache_commit_groups", FEATURE_CACHE_COMMIT_GROUPS) or FEATURE_CACHE_COMMIT_GROUPS))
    delete_group_keys = []
    group_feature_values = []
    group_values = []
    pending_groups = 0
    inserted_features = 0
    worker_count = max(1, int(getattr(args, "feature_cache_workers", FEATURE_CACHE_WORKERS) or 1))
    progress = tqdm(
        iter_group_feature_results(grouped_samples, split_manifest, changed_group_keys, vocab, alias_map, worker_count),
        total=len(changed_group_keys),
        desc="sync feature cache",
        unit="group",
    )
    for result in progress:
        group_key = result["group_key"]
        split = result["split"]
        is_new_group = bool(result["is_new_group"])
        changed = True
        delete_group_keys.append(group_key)
        pick_rows = result["pick_rows"]
        ban_rows = result["ban_rows"]
        group_feature_values.append(
            (
                group_key,
                split,
                1 if is_new_group else 0,
                PREPROCESS_VERSION,
                vocab_hash,
                sqlite3.Binary(pickle.dumps(pick_rows, protocol=pickle.HIGHEST_PROTOCOL)),
                sqlite3.Binary(pickle.dumps(ban_rows, protocol=pickle.HIGHEST_PROTOCOL)),
                sqlite3.Binary(pickle.dumps(result["monster_freq"], protocol=pickle.HIGHEST_PROTOCOL)),
                sqlite3.Binary(pickle.dumps(result["pair_freq"], protocol=pickle.HIGHEST_PROTOCOL)),
                now_value,
            )
        )
        group_values.append(
            (
                group_key,
                split,
                int(result["sample_count"]),
                result["sample_hash"],
                1 if is_new_group else 0,
                PREPROCESS_VERSION,
                vocab_hash,
                now_value,
            )
        )
        updated_groups += 1
        pending_groups += 1
        inserted_features += len(pick_rows) + len(ban_rows) + (1 if split == "train" else 0)
        if pending_groups >= commit_groups:
            commit_feature_cache_batch(conn, delete_group_keys, group_feature_values, group_values)
            delete_group_keys.clear()
            group_feature_values.clear()
            group_values.clear()
            pending_groups = 0
            progress.set_postfix(features=inserted_features, commit_groups=commit_groups, workers=worker_count)

    commit_feature_cache_batch(conn, delete_group_keys, group_feature_values, group_values)

    conn.close()
    return {
        "changed": changed,
        "updated_groups": updated_groups,
        "deleted_groups": deleted_groups,
        "forced_rebuild": force_rebuild,
    }


def load_relation_features_from_cache(path, train_new_only=False):
    out = {
        "train_pos": [],
        "val_pos": [],
        "train_ban_pos": [],
        "val_ban_pos": [],
        "monster_freq": Counter(),
        "pair_freq": Counter(),
    }
    conn = open_feature_cache(path)
    try:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM relation_group_features
                WHERE preprocess_version = ?
                """,
                (PREPROCESS_VERSION,),
            ).fetchone()[0]
        )
        cursor = conn.execute(
            """
            SELECT split, is_new_group, pick_rows, ban_rows, monster_freq, pair_freq
            FROM relation_group_features
            WHERE preprocess_version = ?
            ORDER BY group_key
            """,
            (PREPROCESS_VERSION,),
        )
        with tqdm(total=total, desc="load feature cache", unit="row") as progress:
            while True:
                rows = cursor.fetchmany(FEATURE_FETCH_SIZE)
                if not rows:
                    break
                for split, is_new_group, pick_blob, ban_blob, monster_freq_blob, pair_freq_blob in rows:
                    if train_new_only and split == "train" and not bool(is_new_group):
                        continue
                    pick_rows = pickle.loads(pick_blob)
                    ban_rows = pickle.loads(ban_blob)
                    if split == "train":
                        out["train_pos"].extend(pick_rows)
                        out["train_ban_pos"].extend(ban_rows)
                        out["monster_freq"].update(Counter(pickle.loads(monster_freq_blob)))
                        out["pair_freq"].update(Counter(pickle.loads(pair_freq_blob)))
                    elif split == "val":
                        out["val_pos"].extend(pick_rows)
                        out["val_ban_pos"].extend(ban_rows)
                progress.update(len(rows))
        return out
    finally:
        conn.close()


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return MonsterVocab(json.load(f))


def stable_sample_hash(sample):
    payload = {key: value for key, value in sample.items() if not str(key).startswith("_")}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _stable_slug_list(values):
    return [slugify_fallback(value) for value in (values or []) if value]


def get_split_group_key(sample):
    match_id = str(sample.get("match_id") or "").strip()
    if match_id:
        return f"match_id:{match_id}"

    payload = {
        "first_pick_side": normalize_name(sample.get("first_pick_side")),
        "final_my_pick_slugs": sample.get("final_my_pick_slugs") or _stable_slug_list(sample.get("final_my_picks")),
        "final_enemy_pick_slugs": sample.get("final_enemy_pick_slugs") or _stable_slug_list(sample.get("final_enemy_picks")),
        "my_ban_slug": normalize_name(sample.get("my_ban_slug")) or slugify_fallback(sample.get("my_ban")),
        "enemy_ban_slug": normalize_name(sample.get("enemy_ban_slug")) or slugify_fallback(sample.get("enemy_ban")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"fallback:{hashlib.sha1(raw.encode('utf-8')).hexdigest()}"


def load_split_manifest(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None
    if int(manifest.get("version", 0)) != SPLIT_MANIFEST_VERSION:
        return None
    manifest.setdefault("groups", {})
    return manifest


def save_split_manifest(manifest, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _choose_split_for_new_group(train_count, val_count, val_ratio, rng):
    total_after = train_count + val_count + 1
    val_gap_if_train = abs(val_count / max(1, total_after) - val_ratio)
    val_gap_if_val = abs((val_count + 1) / max(1, total_after) - val_ratio)
    if val_gap_if_val < val_gap_if_train:
        return "val"
    if val_gap_if_val > val_gap_if_train:
        return "train"
    return "val" if rng.random() < val_ratio else "train"


def split_train_val_incremental(samples, val_ratio, seed, manifest_path, rebuild=False):
    manifest = None if rebuild else load_split_manifest(manifest_path)
    manifest_created = manifest is None
    run_id = time.strftime("%Y%m%d%H%M%S")
    if manifest is None:
        manifest = {
            "version": SPLIT_MANIFEST_VERSION,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "split_key": "match_id_or_fallback_hash",
            "groups": {},
        }

    grouped_samples = {}
    for sample in tqdm(samples, desc="split group samples", unit="sample"):
        group_key = get_split_group_key(sample)
        grouped_samples.setdefault(group_key, []).append(sample)

    existing_keys = [key for key in grouped_samples if key in manifest["groups"]]
    new_keys = [key for key in grouped_samples if key not in manifest["groups"]]
    train_group_count = sum(1 for key in existing_keys if manifest["groups"][key].get("split") == "train")
    val_group_count = sum(1 for key in existing_keys if manifest["groups"][key].get("split") == "val")

    rng = random.Random(seed)
    ordered_new_keys = sorted(new_keys)
    rng.shuffle(ordered_new_keys)
    new_group_set = set()

    for group_key in tqdm(ordered_new_keys, desc="assign new groups", unit="group", disable=not ordered_new_keys):
        split = _choose_split_for_new_group(train_group_count, val_group_count, val_ratio, rng)
        manifest["groups"][group_key] = {
            "split": split,
            "sample_count": 0,
            "sample_hashes": [],
            "first_seen_run_id": run_id,
            "last_seen_run_id": run_id,
            "is_new_this_run": True,
        }
        new_group_set.add(group_key)
        if split == "train":
            train_group_count += 1
        else:
            val_group_count += 1

    for group_key, group_samples in tqdm(grouped_samples.items(), desc="update split manifest", unit="group"):
        group_entry = manifest["groups"].setdefault(
            group_key,
            {
                "split": "train",
                "sample_count": 0,
                "sample_hashes": [],
                "first_seen_run_id": run_id,
                "last_seen_run_id": run_id,
                "is_new_this_run": False,
            },
        )
        group_entry["sample_count"] = len(group_samples)
        group_entry["sample_hashes"] = sorted(stable_sample_hash(sample) for sample in group_samples)
        group_entry["last_seen_run_id"] = run_id
        group_entry["is_new_this_run"] = group_key in new_group_set

    for group_key, group_entry in manifest["groups"].items():
        if group_key not in grouped_samples:
            group_entry["is_new_this_run"] = False

    save_split_manifest(manifest, manifest_path)

    train_samples = []
    val_samples = []
    current_train_groups = set()
    current_val_groups = set()
    new_train_groups = set()
    old_train_groups = set()
    new_train_samples = 0
    old_train_samples = 0

    for group_key, group_samples in tqdm(grouped_samples.items(), desc="apply split", unit="group"):
        split = manifest["groups"][group_key]["split"]
        is_new_group = group_key in new_group_set
        for sample in group_samples:
            sample["_split_group_key"] = group_key
            sample["_is_new_group"] = bool(is_new_group)
        if split == "train":
            train_samples.extend(group_samples)
            current_train_groups.add(group_key)
            if is_new_group:
                new_train_groups.add(group_key)
                new_train_samples += len(group_samples)
            else:
                old_train_groups.add(group_key)
                old_train_samples += len(group_samples)
        else:
            val_samples.extend(group_samples)
            current_val_groups.add(group_key)

    split_info = {
        "manifest_created": manifest_created,
        "manifest_path": manifest_path,
        "old_groups_kept": len(existing_keys),
        "new_groups_assigned": len(new_keys),
        "train_group_count": len(current_train_groups),
        "val_group_count": len(current_val_groups),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "new_train_group_count": len(new_train_groups),
        "old_train_group_count": len(old_train_groups),
        "new_train_sample_count": new_train_samples,
        "old_train_sample_count": old_train_samples,
    }
    return train_samples, val_samples, split_info


def extract_all_slugs_from_sample(sample, alias_map=None):
    out = []
    slug_list_keys = [
        "my_picks_slugs",
        "enemy_picks_slugs",
        "remaining_my_pick_slugs",
        "remaining_enemy_pick_slugs",
        "final_my_pick_slugs",
        "final_enemy_pick_slugs",
    ]
    slug_value_keys = [
        "my_leader_slug",
        "enemy_leader_slug",
        "my_ban_slug",
        "enemy_ban_slug",
        "next_pick_slug",
    ]
    fallback_list_keys = [
        "my_picks",
        "enemy_picks",
        "remaining_my_picks",
        "remaining_enemy_picks",
        "final_my_picks",
        "final_enemy_picks",
    ]
    fallback_value_keys = [
        "my_leader",
        "enemy_leader",
        "my_ban",
        "enemy_ban",
        "next_pick",
    ]

    for key in slug_list_keys:
        for monster in sample.get(key, []) or []:
            slug = normalize_name(monster)
            if slug:
                out.append(slug)
    for key in slug_value_keys:
        value = sample.get(key)
        if value:
            slug = normalize_name(value)
            if slug:
                out.append(slug)
    for key in fallback_list_keys:
        for value in sample.get(key, []) or []:
            out.append(fast_resolve_monster_slug(value, alias_map=alias_map))
    for key in fallback_value_keys:
        value = sample.get(key)
        if value:
            out.append(fast_resolve_monster_slug(value, alias_map=alias_map))
    return [slug for slug in out if slug]


def fast_resolve_monster_slug(value, alias_map=None):
    normalized = normalize_name(value)
    if not normalized:
        return ""
    fallback = slugify_fallback(value)
    sanitized_fallback = fallback.replace("[", "").replace("]", "")
    if alias_map:
        for key in (normalized, fallback, sanitized_fallback):
            resolved = alias_map.get(key)
            if resolved:
                return normalize_name(resolved)
    return sanitized_fallback or fallback or normalized


def iter_changed_group_samples(grouped_samples, group_keys):
    for group_key in sorted(group_keys):
        for sample in grouped_samples.get(group_key, []):
            yield sample


def build_or_extend_vocab(
    samples,
    alias_map=None,
    existing_vocab_path=None,
    fine_tune=True,
    grouped_samples=None,
    changed_group_keys=None,
):
    if fine_tune and existing_vocab_path and os.path.exists(existing_vocab_path):
        vocab = load_vocab(existing_vocab_path)
        if grouped_samples is not None and changed_group_keys is not None:
            sample_source = iter_changed_group_samples(grouped_samples, changed_group_keys)
            sample_count = sum(len(grouped_samples.get(group_key, [])) for group_key in changed_group_keys)
        else:
            sample_source = samples
            sample_count = len(samples)
    else:
        vocab = MonsterVocab()
        sample_source = samples
        sample_count = len(samples)
    for sample in tqdm(sample_source, total=sample_count, desc="build vocab", unit="sample"):
        for slug in extract_all_slugs_from_sample(sample, alias_map=alias_map):
            vocab.add_token(slug)
    return vocab


def normalize_final_pick_slugs(sample, my_side=True, alias_map=None):
    slug_key = "final_my_pick_slugs" if my_side else "final_enemy_pick_slugs"
    fallback_key = "final_my_picks" if my_side else "final_enemy_picks"
    return normalize_slug_list(sample.get(slug_key) or sample.get(fallback_key) or [], alias_map=alias_map)


def extract_current_turn_pick_slugs(sample, side, pick_count, alias_map=None):
    pick_count = int(pick_count or 0)
    if pick_count <= 0:
        return []
    next_pick_slug = resolve_monster_slug(sample.get("next_pick_slug"), alias_map=alias_map)
    if not next_pick_slug:
        next_pick_slug = resolve_monster_slug(sample.get("next_pick"), alias_map=alias_map)
    remaining_key = "remaining_my_pick_slugs" if side == "me" else "remaining_enemy_pick_slugs"
    fallback_key = "remaining_my_picks" if side == "me" else "remaining_enemy_picks"
    remaining_picks = normalize_slug_list(
        sample.get(remaining_key) or sample.get(fallback_key) or [],
        alias_map=alias_map,
    )
    targets = [next_pick_slug] if next_pick_slug else []
    for slug in remaining_picks:
        if slug not in targets:
            targets.append(slug)
        if len(targets) >= pick_count:
            break
    return targets[:pick_count] if len(targets) >= pick_count else []


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_result_label(sample):
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


def canonical_pair(a, b):
    return tuple(sorted((int(a), int(b))))


def encode_list(slugs, vocab, max_len=5):
    ids = [vocab.get_id(x) for x in (slugs or [])[:max_len]]
    mask = [1.0] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab.get_id(PAD_TOKEN))
        mask.append(0.0)
    return ids, mask


def collect_frequency_stats(samples, vocab, alias_map, show_progress=True):
    monster_freq = Counter()
    pair_freq = Counter()
    seen_match_comps = set()
    for sample in tqdm(samples, desc="collect frequency", unit="sample", disable=not show_progress):
        final_my = normalize_final_pick_slugs(sample, my_side=True, alias_map=alias_map)
        final_enemy = normalize_final_pick_slugs(sample, my_side=False, alias_map=alias_map)
        all_final = [x for x in final_my + final_enemy if vocab.get_id(x) > 1]
        for slug in all_final:
            monster_freq[vocab.get_id(slug)] += 1

        key = (str(sample.get("match_id") or ""), tuple(sorted(set(final_my))))
        if key in seen_match_comps:
            continue
        seen_match_comps.add(key)
        ids = sorted({vocab.get_id(x) for x in final_my if vocab.get_id(x) > 1})
        for i, first_id in enumerate(ids):
            for second_id in ids[i + 1 :]:
                pair_freq[canonical_pair(first_id, second_id)] += 1
    return monster_freq, pair_freq


def build_positive_actions(samples, vocab, alias_map):
    positives = []
    for sample in samples:
        current_side = normalize_name(sample.get("current_side"))
        if current_side not in {"me", "enemy"}:
            continue

        common_meta = {
            "split_group_key": sample.get("_split_group_key"),
            "is_new_group": bool(sample.get("_is_new_group", False)),
            "sample_weight": float(sample.get("sample_weight", 1.0) or 1.0),
        }
        my_picks = normalize_slug_list(
            sample.get("my_picks_slugs") or sample.get("my_picks") or [],
            alias_map=alias_map,
        )
        enemy_picks = normalize_slug_list(
            sample.get("enemy_picks_slugs") or sample.get("enemy_picks") or [],
            alias_map=alias_map,
        )
        own_picks = my_picks if current_side == "me" else enemy_picks
        opponent_picks = enemy_picks if current_side == "me" else my_picks

        turn = int(sample.get("turn", len(my_picks) + len(enemy_picks) + 1) or 0)
        global_first_pick_side = normalize_name(sample.get("first_pick_side"))
        first_pick_side = 1.0 if global_first_pick_side == current_side else 0.0
        relative_first_pick_side = "me" if first_pick_side > 0.5 else "enemy"
        pick_count = infer_pick_count(own_picks, opponent_picks, relative_first_pick_side)
        current_turn = extract_current_turn_pick_slugs(
            sample,
            current_side,
            pick_count,
            alias_map=alias_map,
        )
        prefix_block = {
            vocab.get_id(x)
            for x in own_picks + opponent_picks
            if vocab.get_id(x) > 1
        }
        result_label = parse_result_label(sample)
        outcome = None if result_label is None else (
            result_label if current_side == "me" else 1.0 - result_label
        )

        if current_turn:
            first_id = vocab.get_id(current_turn[0])
            if first_id > 1:
                positives.append(
                    {
                        "my_picks": own_picks,
                        "enemy_picks": opponent_picks,
                        "candidate": (first_id, vocab.get_id(PAD_TOKEN)),
                        "is_pair": False,
                        "turn": turn,
                        "first_pick_side": first_pick_side,
                        "current_side": 1.0,
                        "outcome": outcome,
                        "blocked": prefix_block,
                        "source": "chronological_prefix_perspective",
                        **common_meta,
                    }
                )

        if pick_count >= 2 and len(current_turn) >= 2:
            first_id = vocab.get_id(current_turn[0])
            second_id = vocab.get_id(current_turn[1])
            if first_id > 1 and second_id > 1 and first_id != second_id:
                positives.append(
                    {
                        "my_picks": own_picks,
                        "enemy_picks": opponent_picks,
                        "candidate": canonical_pair(first_id, second_id),
                        "is_pair": True,
                        "turn": turn,
                        "first_pick_side": first_pick_side,
                        "current_side": 1.0,
                        "outcome": outcome,
                        "blocked": prefix_block,
                        "source": "chronological_prefix_perspective",
                        **common_meta,
                    }
                )
    return positives


def build_ban_positive_actions(samples, vocab, alias_map):
    positives = []
    seen = set()
    for sample in samples:
        final_my = normalize_final_pick_slugs(sample, my_side=True, alias_map=alias_map)
        final_enemy = normalize_final_pick_slugs(sample, my_side=False, alias_map=alias_map)
        if len(final_my) != 5 or len(final_enemy) != 5:
            continue

        enemy_set = set(final_enemy)
        ban_slug = resolve_monster_slug(sample.get("my_ban_slug") or sample.get("my_ban"), alias_map=alias_map)
        if ban_slug not in enemy_set:
            ban_slug = resolve_monster_slug(sample.get("enemy_ban_slug") or sample.get("enemy_ban"), alias_map=alias_map)
        if not ban_slug or ban_slug not in enemy_set:
            continue

        ban_id = vocab.get_id(ban_slug)
        enemy_ids = [vocab.get_id(x) for x in final_enemy if vocab.get_id(x) > 1]
        if ban_id <= 1 or ban_id not in enemy_ids or len(set(enemy_ids)) < 2:
            continue

        match_id = str(sample.get("match_id") or "")
        key = (match_id, tuple(final_my), tuple(final_enemy), ban_slug)
        if key in seen:
            continue
        seen.add(key)

        positives.append(
            {
                "my_picks": final_my[:5],
                "enemy_picks": final_enemy[:5],
                "candidate": ban_id,
                "enemy_candidate_ids": enemy_ids,
                "turn": 10,
                "first_pick_side": 1.0 if normalize_name(sample.get("first_pick_side")) == "me" else 0.0,
                "outcome": parse_result_label(sample),
                "split_group_key": sample.get("_split_group_key"),
                "is_new_group": bool(sample.get("_is_new_group", False)),
                "sample_weight": float(sample.get("sample_weight", 1.0) or 1.0),
            }
        )
    return positives


class RelationPairwiseDataset(Dataset):
    def __init__(self, positives, vocab, monster_freq, pair_freq, negatives_per_positive=4, seed=42):
        self.vocab = vocab
        self.monster_freq = Counter(monster_freq)
        self.pair_freq = Counter(pair_freq)
        self.negatives_per_positive = int(negatives_per_positive)
        self.seed = int(seed)
        self.pad_id = vocab.get_id(PAD_TOKEN)
        self.valid_ids = [i for i in range(len(vocab)) if i > 1]
        self.high_freq_ids = [idx for idx, _ in self.monster_freq.most_common()] or self.valid_ids
        self.max_action_freq = max([1] + list(self.monster_freq.values()) + list(self.pair_freq.values()))

        freq_items = sorted(((idx, self.monster_freq.get(idx, 0)) for idx in self.valid_ids), key=lambda x: x[1])
        self.freq_rank = {idx: pos for pos, (idx, _) in enumerate(freq_items)}
        self.sorted_by_freq = [idx for idx, _ in freq_items]
        self.base_rows = [self._prepare_positive(positive) for positive in positives]
        self.base_count = len(self.base_rows)
        self.row_count = self.base_count * self.negatives_per_positive
        self.row_base_indices = [idx for idx in range(self.base_count) for _ in range(self.negatives_per_positive)]
        self.sample_is_new_group = [1 if self.base_rows[idx]["is_new_group"] else 0 for idx in self.row_base_indices]
        self.last_refresh_seconds = 0.0
        self._build_static_tensors()
        self.refresh_negative_pairs(self.seed)

    def _prepare_positive(self, positive):
        row = dict(positive)
        my_ids, my_mask = encode_list(row["my_picks"], self.vocab)
        enemy_ids, enemy_mask = encode_list(row["enemy_picks"], self.vocab)
        row["my_ids"] = my_ids
        row["my_mask"] = my_mask
        row["enemy_ids"] = enemy_ids
        row["enemy_mask"] = enemy_mask
        row["turn_value"] = float(row["turn"])
        row["first_pick_side_value"] = float(row["first_pick_side"])
        row["current_side_value"] = float(row.get("current_side", 1.0))
        row["is_new_group"] = bool(row.get("is_new_group", False))
        row["sample_weight"] = max(1e-6, float(row.get("sample_weight", 1.0) or 1.0))
        row["outcome_value"] = float(row["outcome"]) if row.get("outcome") is not None else 0.5
        row["outcome_mask"] = 1.0 if row.get("outcome") is not None else 0.0
        row["candidate"] = (int(row["candidate"][0]), int(row["candidate"][1]))
        row["blocked_ids"] = {int(idx) for idx in row.get("blocked", set())}
        row["legal_ids"] = [idx for idx in self.valid_ids if idx not in row["blocked_ids"]]
        return row

    def _build_static_tensors(self):
        if self.row_count == 0:
            empty_long = torch.empty((0,), dtype=torch.long)
            empty_float = torch.empty((0,), dtype=torch.float32)
            self.my_picks_tensor = torch.empty((0, 5), dtype=torch.long)
            self.my_mask_tensor = torch.empty((0, 5), dtype=torch.float32)
            self.enemy_picks_tensor = torch.empty((0, 5), dtype=torch.long)
            self.enemy_mask_tensor = torch.empty((0, 5), dtype=torch.float32)
            self.pos_c1_tensor = empty_long.clone()
            self.pos_c2_tensor = empty_long.clone()
            self.neg_c1_tensor = empty_long.clone()
            self.neg_c2_tensor = empty_long.clone()
            self.action_mask_tensor = torch.empty((0, 2), dtype=torch.float32)
            self.action_type_tensor = empty_long.clone()
            self.turn_tensor = empty_float.clone()
            self.first_pick_side_tensor = empty_float.clone()
            self.current_side_tensor = empty_float.clone()
            self.loss_weight_tensor = empty_float.clone()
            self.outcome_tensor = empty_float.clone()
            self.outcome_mask_tensor = empty_float.clone()
            self.is_pair_tensor = empty_long.clone()
            self.is_new_group_tensor = empty_long.clone()
            return

        self.my_picks_tensor = torch.tensor(
            [self.base_rows[idx]["my_ids"] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.my_mask_tensor = torch.tensor(
            [self.base_rows[idx]["my_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.enemy_picks_tensor = torch.tensor(
            [self.base_rows[idx]["enemy_ids"] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.enemy_mask_tensor = torch.tensor(
            [self.base_rows[idx]["enemy_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.pos_c1_tensor = torch.tensor(
            [self.base_rows[idx]["candidate"][0] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.pos_c2_tensor = torch.tensor(
            [self.base_rows[idx]["candidate"][1] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.action_mask_tensor = torch.tensor(
            [[1.0, 1.0 if self.base_rows[idx]["is_pair"] else 0.0] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.action_type_tensor = torch.tensor(
            [1 if self.base_rows[idx]["is_pair"] else 0 for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.turn_tensor = torch.tensor(
            [self.base_rows[idx]["turn_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.first_pick_side_tensor = torch.tensor(
            [self.base_rows[idx]["first_pick_side_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.current_side_tensor = torch.tensor(
            [self.base_rows[idx]["current_side_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.is_pair_tensor = torch.tensor(
            [1 if self.base_rows[idx]["is_pair"] else 0 for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.is_new_group_tensor = torch.tensor(self.sample_is_new_group, dtype=torch.long)
        self.neg_c1_tensor = torch.empty((self.row_count,), dtype=torch.long)
        self.neg_c2_tensor = torch.empty((self.row_count,), dtype=torch.long)
        self.loss_weight_tensor = torch.empty((self.row_count,), dtype=torch.float32)
        self.outcome_tensor = torch.tensor(
            [self.base_rows[idx]["outcome_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.outcome_mask_tensor = torch.tensor(
            [self.base_rows[idx]["outcome_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )

    def __len__(self):
        return self.row_count

    def _near_frequency_id(self, positive_id, legal, rng):
        if not legal:
            return None
        rank = self.freq_rank.get(int(positive_id), len(self.sorted_by_freq) // 2)
        window = max(8, len(self.sorted_by_freq) // 20)
        candidates = [
            idx
            for idx in self.sorted_by_freq[max(0, rank - window) : min(len(self.sorted_by_freq), rank + window + 1)]
            if idx in legal
        ]
        return rng.choice(candidates or list(legal))

    def _sample_single_negative(self, positive, rng):
        positive_id = int(positive["candidate"][0])
        legal = set(positive["legal_ids"])
        legal.discard(positive_id)
        if not legal:
            return positive_id

        roll = rng.random()
        if roll < HIGH_FREQ_NEGATIVE_RATIO:
            pool = [idx for idx in self.high_freq_ids if idx in legal]
            return rng.choice(pool or sorted(legal))
        if roll < HIGH_FREQ_NEGATIVE_RATIO + NEAR_FREQ_NEGATIVE_RATIO:
            return self._near_frequency_id(positive_id, legal, rng)
        return rng.choice(sorted(legal))

    def _sample_pair_negative(self, positive, rng):
        pos_pair = canonical_pair(*positive["candidate"])
        legal = set(positive["legal_ids"])
        if len(legal) < 2:
            return pos_pair

        def valid_pair(pair):
            return pair[0] != pair[1] and pair != pos_pair and pair[0] in legal and pair[1] in legal

        roll = rng.random()
        for _ in range(64):
            if roll < HIGH_FREQ_NEGATIVE_RATIO:
                pool = [idx for idx in self.high_freq_ids if idx in legal]
                if len(pool) < 2:
                    pool = sorted(legal)
                pair = canonical_pair(*rng.sample(pool, 2))
            elif roll < HIGH_FREQ_NEGATIVE_RATIO + NEAR_FREQ_NEGATIVE_RATIO:
                keep = rng.choice(pos_pair)
                other_pos = pos_pair[0] if keep == pos_pair[1] else pos_pair[1]
                replacement = self._near_frequency_id(other_pos, legal - {keep}, rng)
                pair = canonical_pair(keep, replacement)
            else:
                pair = canonical_pair(*rng.sample(sorted(legal), 2))
            if valid_pair(pair):
                return pair
        return canonical_pair(*rng.sample(sorted(legal), 2))

    def _action_freq(self, candidate, is_pair):
        if is_pair:
            pair = canonical_pair(*candidate)
            if self.pair_freq.get(pair, 0) > 0:
                return float(self.pair_freq[pair])
            return math.sqrt(float(self.monster_freq.get(pair[0], 0) * self.monster_freq.get(pair[1], 0)))
        return float(self.monster_freq.get(int(candidate[0]), 0))

    def _hard_candidate_pool(self, positive, rng, candidate_cap):
        legal = set(positive["legal_ids"])
        cap = max(4, int(candidate_cap))
        if positive["is_pair"]:
            pos_pair = canonical_pair(*positive["candidate"])
            pool_size = max(8, min(32, int(math.sqrt(cap * 2)) + 4))
            pool = [idx for idx in self.high_freq_ids if idx in legal][:pool_size]
            remainder = sorted(legal - set(pool))
            if len(pool) < pool_size and remainder:
                pool.extend(rng.sample(remainder, min(pool_size - len(pool), len(remainder))))
            for idx in pos_pair:
                if idx in legal and idx not in pool:
                    pool.append(idx)
            pairs = [
                pair
                for pair in combinations(sorted(set(pool)), 2)
                if pair != pos_pair and pair[0] in legal and pair[1] in legal
            ]
            if len(pairs) > cap:
                pairs = rng.sample(pairs, cap)
            return pairs

        positive_id = int(positive["candidate"][0])
        high = [idx for idx in self.high_freq_ids if idx in legal and idx != positive_id]
        pool = high[:cap]
        remainder = sorted(legal - set(pool) - {positive_id})
        if len(pool) < cap and remainder:
            pool.extend(rng.sample(remainder, min(cap - len(pool), len(remainder))))
        return [(idx, self.pad_id) for idx in pool if idx != positive_id]

    @torch.no_grad()
    def _mine_hard_negatives(self, model, device, base_indices, rng, candidate_cap, batch_size=1024):
        if model is None or not base_indices:
            return {}
        tasks = []
        for base_idx in base_indices:
            positive = self.base_rows[base_idx]
            tasks.extend(
                (base_idx, positive, candidate)
                for candidate in self._hard_candidate_pool(positive, rng, candidate_cap)
            )
        if not tasks:
            return {}

        model_was_training = model.training
        model.eval()
        best_scores = {}
        best_candidates = {}
        for start in range(0, len(tasks), max(1, int(batch_size))):
            chunk = tasks[start : start + max(1, int(batch_size))]
            scores = model(
                torch.tensor([item[1]["my_ids"] for item in chunk], dtype=torch.long, device=device),
                torch.tensor([item[1]["my_mask"] for item in chunk], dtype=torch.float32, device=device),
                torch.tensor([item[1]["enemy_ids"] for item in chunk], dtype=torch.long, device=device),
                torch.tensor([item[1]["enemy_mask"] for item in chunk], dtype=torch.float32, device=device),
                torch.tensor([item[2][0] for item in chunk], dtype=torch.long, device=device),
                torch.tensor([item[2][1] for item in chunk], dtype=torch.long, device=device),
                torch.tensor(
                    [[1.0, 1.0 if item[1]["is_pair"] else 0.0] for item in chunk],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.tensor([item[1]["turn_value"] for item in chunk], dtype=torch.float32, device=device),
                torch.tensor(
                    [item[1]["first_pick_side_value"] for item in chunk],
                    dtype=torch.float32,
                    device=device,
                ),
                action_type=torch.tensor(
                    [1 if item[1]["is_pair"] else 0 for item in chunk],
                    dtype=torch.long,
                    device=device,
                ),
                current_side=torch.tensor(
                    [item[1]["current_side_value"] for item in chunk],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            for task, score in zip(chunk, scores.detach().cpu().tolist()):
                base_idx, _positive, candidate = task
                if base_idx not in best_scores or float(score) > best_scores[base_idx]:
                    best_scores[base_idx] = float(score)
                    best_candidates[base_idx] = candidate
        if model_was_training:
            model.train()
        return best_candidates

    def _debiased_frequency_weight(self, frequency):
        propensity = (float(frequency) + 1.0) / (float(self.max_action_freq) + 1.0)
        return min(2.0, max(0.5, propensity ** -0.25))

    def refresh_negative_pairs(
        self,
        epoch_seed: int,
        model=None,
        device=None,
        hard_negative_ratio=0.0,
        hard_negative_candidate_pool=64,
        hard_negative_max_rows=4096,
    ):
        start = time.perf_counter()
        if self.row_count == 0:
            self.last_refresh_seconds = time.perf_counter() - start
            return

        rng = random.Random(int(epoch_seed))
        hard_negatives = {}
        hard_ratio = min(1.0, max(0.0, float(hard_negative_ratio)))
        if model is not None and device is not None and hard_ratio > 0.0:
            hard_count = min(
                self.base_count,
                max(1, min(int(hard_negative_max_rows), int(self.base_count * hard_ratio))),
            )
            hard_indices = rng.sample(range(self.base_count), hard_count)
            hard_negatives = self._mine_hard_negatives(
                model=model,
                device=device,
                base_indices=hard_indices,
                rng=rng,
                candidate_cap=hard_negative_candidate_pool,
            )

        neg_c1 = []
        neg_c2 = []
        loss_weight = []
        for base_idx in self.row_base_indices:
            positive = self.base_rows[base_idx]
            is_pair = bool(positive["is_pair"])
            if base_idx in hard_negatives:
                negative = hard_negatives[base_idx]
            else:
                negative = (
                    self._sample_pair_negative(positive, rng)
                    if is_pair
                    else (self._sample_single_negative(positive, rng), self.pad_id)
                )
            pos = positive["candidate"]
            pos_freq = self._action_freq(pos, is_pair)
            neg_freq = self._action_freq(negative, is_pair)
            weight = positive["sample_weight"] * (
                self._debiased_frequency_weight(pos_freq) * self._debiased_frequency_weight(neg_freq)
            )
            neg_c1.append(int(negative[0]))
            neg_c2.append(int(negative[1]))
            loss_weight.append(float(weight))

        self.neg_c1_tensor = torch.tensor(neg_c1, dtype=torch.long)
        self.neg_c2_tensor = torch.tensor(neg_c2, dtype=torch.long)
        self.loss_weight_tensor = torch.tensor(loss_weight, dtype=torch.float32)
        self.last_refresh_seconds = time.perf_counter() - start

    def __getitem__(self, idx):
        return {
            "my_picks": self.my_picks_tensor[idx],
            "my_mask": self.my_mask_tensor[idx],
            "enemy_picks": self.enemy_picks_tensor[idx],
            "enemy_mask": self.enemy_mask_tensor[idx],
            "pos_c1": self.pos_c1_tensor[idx],
            "pos_c2": self.pos_c2_tensor[idx],
            "neg_c1": self.neg_c1_tensor[idx],
            "neg_c2": self.neg_c2_tensor[idx],
            "action_mask": self.action_mask_tensor[idx],
            "action_type": self.action_type_tensor[idx],
            "turn": self.turn_tensor[idx],
            "first_pick_side": self.first_pick_side_tensor[idx],
            "current_side": self.current_side_tensor[idx],
            "loss_weight": self.loss_weight_tensor[idx],
            "outcome": self.outcome_tensor[idx],
            "outcome_mask": self.outcome_mask_tensor[idx],
            "is_pair": self.is_pair_tensor[idx],
            "is_new_group": self.is_new_group_tensor[idx],
        }


class BPBanNegativeSampler:
    def __init__(self, ban_freq, seed=42):
        self.ban_freq = Counter(ban_freq)
        self.rng = random.Random(seed)

    def sample(self, positive):
        pos_id = int(positive["candidate"])
        legal = [int(idx) for idx in positive["enemy_candidate_ids"] if int(idx) != pos_id]
        if not legal:
            return pos_id
        roll = self.rng.random()
        if roll < 0.45:
            return max(legal, key=lambda idx: self.ban_freq.get(idx, 0))
        if roll < 0.80:
            target_freq = self.ban_freq.get(pos_id, 0)
            return min(legal, key=lambda idx: abs(self.ban_freq.get(idx, 0) - target_freq))
        return self.rng.choice(legal)


class BanPairwiseDataset(Dataset):
    def __init__(self, positives, vocab, ban_freq, negatives_per_positive=4, seed=42):
        self.vocab = vocab
        self.pad_id = vocab.get_id(PAD_TOKEN)
        self.sampler = BPBanNegativeSampler(ban_freq, seed=seed)
        self.seed = int(seed)
        self.base_rows = [self._prepare_positive(positive) for positive in positives]
        self.negatives_per_positive = int(negatives_per_positive)
        self.base_count = len(self.base_rows)
        self.row_count = self.base_count * self.negatives_per_positive
        self.row_base_indices = [idx for idx in range(self.base_count) for _ in range(self.negatives_per_positive)]
        self.sample_is_new_group = [1 if self.base_rows[idx]["is_new_group"] else 0 for idx in self.row_base_indices]
        self.last_refresh_seconds = 0.0
        self._build_static_tensors()
        self.refresh_negative_pairs(self.seed)

    def _prepare_positive(self, positive):
        row = dict(positive)
        my_ids, my_mask = encode_list(row["my_picks"], self.vocab)
        enemy_ids, enemy_mask = encode_list(row["enemy_picks"], self.vocab)
        row["my_ids"] = my_ids
        row["my_mask"] = my_mask
        row["enemy_ids"] = enemy_ids
        row["enemy_mask"] = enemy_mask
        row["turn_value"] = float(row["turn"])
        row["first_pick_side_value"] = float(row["first_pick_side"])
        row["current_side_value"] = 1.0
        row["is_new_group"] = bool(row.get("is_new_group", False))
        row["sample_weight"] = max(1e-6, float(row.get("sample_weight", 1.0) or 1.0))
        row["outcome_value"] = float(row["outcome"]) if row.get("outcome") is not None else 0.5
        row["outcome_mask"] = 1.0 if row.get("outcome") is not None else 0.0
        row["candidate"] = int(row["candidate"])
        row["legal_enemy_ids"] = [int(idx) for idx in row["enemy_candidate_ids"]]
        return row

    def _build_static_tensors(self):
        if self.row_count == 0:
            empty_long = torch.empty((0,), dtype=torch.long)
            empty_float = torch.empty((0,), dtype=torch.float32)
            self.my_picks_tensor = torch.empty((0, 5), dtype=torch.long)
            self.my_mask_tensor = torch.empty((0, 5), dtype=torch.float32)
            self.enemy_picks_tensor = torch.empty((0, 5), dtype=torch.long)
            self.enemy_mask_tensor = torch.empty((0, 5), dtype=torch.float32)
            self.pos_c1_tensor = empty_long.clone()
            self.pos_c2_tensor = empty_long.clone()
            self.neg_c1_tensor = empty_long.clone()
            self.neg_c2_tensor = empty_long.clone()
            self.action_mask_tensor = torch.empty((0, 2), dtype=torch.float32)
            self.action_type_tensor = empty_long.clone()
            self.turn_tensor = empty_float.clone()
            self.first_pick_side_tensor = empty_float.clone()
            self.current_side_tensor = empty_float.clone()
            self.loss_weight_tensor = empty_float.clone()
            self.outcome_tensor = empty_float.clone()
            self.outcome_mask_tensor = empty_float.clone()
            self.is_pair_tensor = empty_long.clone()
            self.is_new_group_tensor = empty_long.clone()
            return

        self.my_picks_tensor = torch.tensor(
            [self.base_rows[idx]["my_ids"] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.my_mask_tensor = torch.tensor(
            [self.base_rows[idx]["my_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.enemy_picks_tensor = torch.tensor(
            [self.base_rows[idx]["enemy_ids"] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.enemy_mask_tensor = torch.tensor(
            [self.base_rows[idx]["enemy_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.pos_c1_tensor = torch.tensor(
            [self.base_rows[idx]["candidate"] for idx in self.row_base_indices],
            dtype=torch.long,
        )
        self.pos_c2_tensor = torch.full((self.row_count,), self.pad_id, dtype=torch.long)
        self.action_mask_tensor = torch.tensor([[1.0, 0.0]] * self.row_count, dtype=torch.float32)
        self.action_type_tensor = torch.full((self.row_count,), 2, dtype=torch.long)
        self.turn_tensor = torch.tensor(
            [self.base_rows[idx]["turn_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.first_pick_side_tensor = torch.tensor(
            [self.base_rows[idx]["first_pick_side_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.current_side_tensor = torch.tensor(
            [self.base_rows[idx]["current_side_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.loss_weight_tensor = torch.tensor(
            [self.base_rows[idx]["sample_weight"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.outcome_tensor = torch.tensor(
            [self.base_rows[idx]["outcome_value"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.outcome_mask_tensor = torch.tensor(
            [self.base_rows[idx]["outcome_mask"] for idx in self.row_base_indices],
            dtype=torch.float32,
        )
        self.is_pair_tensor = torch.zeros((self.row_count,), dtype=torch.long)
        self.is_new_group_tensor = torch.tensor(self.sample_is_new_group, dtype=torch.long)
        self.neg_c1_tensor = torch.empty((self.row_count,), dtype=torch.long)
        self.neg_c2_tensor = torch.full((self.row_count,), self.pad_id, dtype=torch.long)

    def __len__(self):
        return self.row_count

    def refresh_negative_pairs(self, epoch_seed: int):
        start = time.perf_counter()
        if self.row_count == 0:
            self.last_refresh_seconds = time.perf_counter() - start
            return

        self.sampler.rng.seed(int(epoch_seed))
        neg_c1 = []
        for base_idx in self.row_base_indices:
            positive = self.base_rows[base_idx]
            neg_c1.append(int(self.sampler.sample(positive)))
        self.neg_c1_tensor = torch.tensor(neg_c1, dtype=torch.long)
        self.last_refresh_seconds = time.perf_counter() - start

    def __getitem__(self, idx):
        return {
            "my_picks": self.my_picks_tensor[idx],
            "my_mask": self.my_mask_tensor[idx],
            "enemy_picks": self.enemy_picks_tensor[idx],
            "enemy_mask": self.enemy_mask_tensor[idx],
            "pos_c1": self.pos_c1_tensor[idx],
            "pos_c2": self.pos_c2_tensor[idx],
            "neg_c1": self.neg_c1_tensor[idx],
            "neg_c2": self.neg_c2_tensor[idx],
            "action_mask": self.action_mask_tensor[idx],
            "action_type": self.action_type_tensor[idx],
            "turn": self.turn_tensor[idx],
            "first_pick_side": self.first_pick_side_tensor[idx],
            "current_side": self.current_side_tensor[idx],
            "loss_weight": self.loss_weight_tensor[idx],
            "outcome": self.outcome_tensor[idx],
            "outcome_mask": self.outcome_mask_tensor[idx],
            "is_pair": self.is_pair_tensor[idx],
            "is_new_group": self.is_new_group_tensor[idx],
        }


def collate_fn(batch):
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in batch[0]}


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_loader(dataset, batch_size, sampler=None, shuffle=False, num_workers=0, pin_memory=False, prefetch_factor=4, persistent_workers=False):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": bool(shuffle and sampler is None and len(dataset) > 0),
        "sampler": sampler,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["pin_memory"] = bool(pin_memory)
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
    loss_scale=1.0,
    outcome_loss_weight=OUTCOME_LOSS_WEIGHT,
    desc_prefix="",
    grad_accum_steps=1,
    log_interval=50,
):
    if len(loader.dataset) == 0:
        return {
            "loss": 0.0,
            "overall_acc": 0.0,
            "single_acc": 0.0,
            "pair_acc": 0.0,
        }

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_outcome_loss = 0.0
    total = 0
    correct = 0
    single_correct = 0
    single_total = 0
    pair_correct = 0
    pair_total = 0
    running_loss = 0.0
    iterator = tqdm(loader, desc=f"{desc_prefix}{'train' if is_train else 'val'}", leave=False)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pos_score = model(
                batch["my_picks"],
                batch["my_mask"],
                batch["enemy_picks"],
                batch["enemy_mask"],
                batch["pos_c1"],
                batch["pos_c2"],
                batch["action_mask"],
                batch["turn"],
                batch["first_pick_side"],
                action_type=batch.get("action_type"),
                current_side=batch.get("current_side"),
            )
            neg_score = model(
                batch["my_picks"],
                batch["my_mask"],
                batch["enemy_picks"],
                batch["enemy_mask"],
                batch["neg_c1"],
                batch["neg_c2"],
                batch["action_mask"],
                batch["turn"],
                batch["first_pick_side"],
                action_type=batch.get("action_type"),
                current_side=batch.get("current_side"),
            )
            per_sample = F.softplus(-(pos_score - neg_score))
            metric_loss = (per_sample * batch["loss_weight"]).mean()
            outcome_mask = batch.get("outcome_mask")
            if outcome_mask is not None and bool(torch.any(outcome_mask > 0)):
                outcome_loss_values = F.binary_cross_entropy_with_logits(
                    pos_score,
                    batch["outcome"],
                    reduction="none",
                )
                outcome_loss = (
                    (outcome_loss_values * outcome_mask * batch["loss_weight"]).sum()
                    / (outcome_mask * batch["loss_weight"]).sum().clamp_min(1e-6)
                )
            else:
                outcome_loss = torch.zeros((), dtype=metric_loss.dtype, device=metric_loss.device)
            combined_loss = metric_loss + float(outcome_loss_weight) * outcome_loss
            loss = combined_loss * float(loss_scale)

        if is_train:
            scaled_loss = loss / max(1, int(grad_accum_steps))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = step % max(1, int(grad_accum_steps)) == 0 or step == len(loader)
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = int(batch["is_pair"].numel())
        loss_value = float(metric_loss.detach().item())
        outcome_loss_value = float(outcome_loss.detach().item())
        total_loss += loss_value * batch_size
        total_outcome_loss += outcome_loss_value * batch_size
        total += batch_size
        running_loss += loss_value
        batch_correct = pos_score > neg_score
        correct += int(batch_correct.sum().detach().item())
        single_mask = batch["is_pair"] == 0
        pair_mask = batch["is_pair"] == 1
        single_total += int(single_mask.sum().detach().item())
        pair_total += int(pair_mask.sum().detach().item())
        if single_mask.any():
            single_correct += int(batch_correct[single_mask].sum().detach().item())
        if pair_mask.any():
            pair_correct += int(batch_correct[pair_mask].sum().detach().item())
        if log_interval > 0 and step % log_interval == 0:
            iterator.set_postfix(loss=f"{running_loss / log_interval:.4f}")
            running_loss = 0.0

    return {
        "loss": total_loss / max(1, total),
        "outcome_loss": total_outcome_loss / max(1, total),
        "overall_acc": correct / max(1, total),
        "single_acc": single_correct / max(1, single_total),
        "pair_acc": pair_correct / max(1, pair_total),
    }


@torch.no_grad()
def evaluate_ban_topk(model, positives, vocab, device, batch_size=512):
    if not positives:
        return {"top1_acc": 0.0, "top3_acc": 0.0}

    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    pad_id = vocab.get_id(PAD_TOKEN)

    prepared = []
    for positive in positives:
        row = dict(positive)
        my_ids, my_mask = encode_list(row["my_picks"], vocab)
        enemy_ids, enemy_mask = encode_list(row["enemy_picks"], vocab)
        row["my_ids"] = my_ids
        row["my_mask"] = my_mask
        row["enemy_ids"] = enemy_ids
        row["enemy_mask"] = enemy_mask
        prepared.append(row)

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        expanded = []
        for row_idx, positive in enumerate(chunk):
            for candidate_id in positive["enemy_candidate_ids"]:
                expanded.append((row_idx, positive, int(candidate_id)))
        if not expanded:
            continue

        scores = model(
            torch.tensor([item[1]["my_ids"] for item in expanded], dtype=torch.long, device=device),
            torch.tensor([item[1]["my_mask"] for item in expanded], dtype=torch.float32, device=device),
            torch.tensor([item[1]["enemy_ids"] for item in expanded], dtype=torch.long, device=device),
            torch.tensor([item[1]["enemy_mask"] for item in expanded], dtype=torch.float32, device=device),
            torch.tensor([item[2] for item in expanded], dtype=torch.long, device=device),
            torch.full((len(expanded),), pad_id, dtype=torch.long, device=device),
            torch.tensor([[1.0, 0.0]] * len(expanded), dtype=torch.float32, device=device),
            torch.tensor([10.0] * len(expanded), dtype=torch.float32, device=device),
            torch.tensor([float(item[1]["first_pick_side"]) for item in expanded], dtype=torch.float32, device=device),
            action_type=torch.full((len(expanded),), 2, dtype=torch.long, device=device),
            current_side=torch.ones((len(expanded),), dtype=torch.float32, device=device),
        )
        grouped = {}
        for (row_idx, positive, candidate_id), score in zip(expanded, scores.detach().cpu().tolist()):
            grouped.setdefault(row_idx, []).append((float(score), candidate_id, int(positive["candidate"])))
        for rows in grouped.values():
            rows.sort(key=lambda item: item[0], reverse=True)
            positive_id = rows[0][2]
            total += 1
            if rows[0][1] == positive_id:
                top1 += 1
            if positive_id in {candidate_id for _, candidate_id, _ in rows[:3]}:
                top3 += 1

    return {
        "top1_acc": top1 / max(1, total),
        "top3_acc": top3 / max(1, total),
    }


@torch.no_grad()
def evaluate_pick_topk(
    model,
    positives,
    vocab,
    device,
    candidate_cap=96,
    max_samples=2048,
    batch_size=1024,
):
    if not positives:
        return {
            "single_top1_acc": 0.0,
            "single_top3_acc": 0.0,
            "pair_top1_acc": 0.0,
            "pair_top3_acc": 0.0,
            "single_total": 0,
            "pair_total": 0,
        }

    model.eval()
    valid_ids = [idx for idx in range(len(vocab)) if idx > 1]
    pad_id = vocab.get_id(PAD_TOKEN)
    if max_samples > 0 and len(positives) > int(max_samples):
        sample_rng = random.Random(0)
        selected_indices = sorted(sample_rng.sample(range(len(positives)), int(max_samples)))
        positives = [positives[idx] for idx in selected_indices]

    eval_rows = []
    for positive in positives:
        is_pair = bool(positive.get("is_pair"))
        positive_action = (
            canonical_pair(*positive["candidate"])
            if is_pair
            else (int(positive["candidate"][0]), pad_id)
        )
        blocked = {int(idx) for idx in positive.get("blocked", set())}
        legal = [idx for idx in valid_ids if idx not in blocked]
        if is_pair:
            positive_ids = set(positive_action)
            candidate_ids = list(dict.fromkeys(
                list(positive_ids) + [idx for idx in legal if idx not in positive_ids]
            ))
            if candidate_cap > 0:
                pool_limit = max(4, int(math.sqrt(max(4, int(candidate_cap)) * 2)) + 1)
                candidate_ids = candidate_ids[:pool_limit]
            actions = [
                canonical_pair(first_id, second_id)
                for first_id, second_id in combinations(candidate_ids, 2)
                if canonical_pair(first_id, second_id) != positive_action
            ]
            if candidate_cap > 0 and len(actions) > int(candidate_cap) - 1:
                actions = actions[: max(1, int(candidate_cap) - 1)]
            actions.insert(0, positive_action)
        else:
            candidate_ids = [int(positive_action[0])] + [
                idx for idx in legal if idx != positive_action[0]
            ]
            if candidate_cap > 0:
                candidate_ids = candidate_ids[: max(1, int(candidate_cap))]
            actions = [(idx, pad_id) for idx in candidate_ids]
            if positive_action not in actions:
                actions.insert(0, positive_action)

        if actions:
            eval_rows.append((positive, actions, positive_action))

    expanded = []
    for row_idx, (positive, actions, positive_action) in enumerate(eval_rows):
        my_ids, my_mask = encode_list(positive["my_picks"], vocab)
        enemy_ids, enemy_mask = encode_list(positive["enemy_picks"], vocab)
        for action in actions:
            expanded.append(
                {
                    "row_idx": row_idx,
                    "my_ids": my_ids,
                    "my_mask": my_mask,
                    "enemy_ids": enemy_ids,
                    "enemy_mask": enemy_mask,
                    "candidate": action,
                    "is_pair": bool(positive.get("is_pair")),
                    "turn": float(positive["turn"]),
                    "first_pick_side": float(positive["first_pick_side"]),
                    "current_side": float(positive.get("current_side", 1.0)),
                }
            )

    grouped_scores = {}
    for start in range(0, len(expanded), max(1, int(batch_size))):
        chunk = expanded[start : start + max(1, int(batch_size))]
        scores = model(
            torch.tensor([row["my_ids"] for row in chunk], dtype=torch.long, device=device),
            torch.tensor([row["my_mask"] for row in chunk], dtype=torch.float32, device=device),
            torch.tensor([row["enemy_ids"] for row in chunk], dtype=torch.long, device=device),
            torch.tensor([row["enemy_mask"] for row in chunk], dtype=torch.float32, device=device),
            torch.tensor([row["candidate"][0] for row in chunk], dtype=torch.long, device=device),
            torch.tensor([row["candidate"][1] for row in chunk], dtype=torch.long, device=device),
            torch.tensor(
                [[1.0, 1.0 if row["is_pair"] else 0.0] for row in chunk],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor([row["turn"] for row in chunk], dtype=torch.float32, device=device),
            torch.tensor(
                [row["first_pick_side"] for row in chunk],
                dtype=torch.float32,
                device=device,
            ),
            action_type=torch.tensor(
                [1 if row["is_pair"] else 0 for row in chunk],
                dtype=torch.long,
                device=device,
            ),
            current_side=torch.tensor(
                [row["current_side"] for row in chunk],
                dtype=torch.float32,
                device=device,
            ),
        )
        for row, score in zip(chunk, scores.detach().cpu().tolist()):
            grouped_scores.setdefault(row["row_idx"], []).append((float(score), row["candidate"]))

    metrics = {
        "single_top1_acc": 0.0,
        "single_top3_acc": 0.0,
        "pair_top1_acc": 0.0,
        "pair_top3_acc": 0.0,
        "single_total": 0,
        "pair_total": 0,
    }
    for row_idx, rows in grouped_scores.items():
        rows.sort(key=lambda item: item[0], reverse=True)
        positive_action = eval_rows[row_idx][2]
        is_pair = bool(eval_rows[row_idx][0].get("is_pair"))
        prefix = "pair" if is_pair else "single"
        metrics[f"{prefix}_total"] += 1
        if rows and rows[0][1] == positive_action:
            metrics[f"{prefix}_top1_acc"] += 1.0
        if positive_action in {candidate for _score, candidate in rows[:3]}:
            metrics[f"{prefix}_top3_acc"] += 1.0

    for prefix in ("single", "pair"):
        total = max(1, int(metrics[f"{prefix}_total"]))
        metrics[f"{prefix}_top1_acc"] /= total
        metrics[f"{prefix}_top3_acc"] /= total
    return metrics


@torch.no_grad()
def fit_relation_score_calibration(model, loader, device, max_values=200000):
    model.eval()
    values = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pos_score = model(
            batch["my_picks"],
            batch["my_mask"],
            batch["enemy_picks"],
            batch["enemy_mask"],
            batch["pos_c1"],
            batch["pos_c2"],
            batch["action_mask"],
            batch["turn"],
            batch["first_pick_side"],
            action_type=batch.get("action_type"),
            current_side=batch.get("current_side"),
        )
        neg_score = model(
            batch["my_picks"],
            batch["my_mask"],
            batch["enemy_picks"],
            batch["enemy_mask"],
            batch["neg_c1"],
            batch["neg_c2"],
            batch["action_mask"],
            batch["turn"],
            batch["first_pick_side"],
            action_type=batch.get("action_type"),
            current_side=batch.get("current_side"),
        )
        values.extend(pos_score.detach().cpu().tolist())
        values.extend(neg_score.detach().cpu().tolist())
        if len(values) >= int(max_values):
            break
    if not values:
        return {"center": 0.0, "scale": 1.0}
    scores = torch.tensor(values[: int(max_values)], dtype=torch.float32)
    return {
        "center": float(scores.mean().item()),
        "scale": max(0.5, float(scores.std(unbiased=False).item())),
    }


def load_relation_checkpoint_for_finetune(model, model_path):
    if not model_path or not os.path.exists(model_path):
        return False
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    current = model.state_dict()
    adapted = {}
    loaded = []
    partial = []
    skipped = []

    for key, value in checkpoint.items():
        if key not in current:
            skipped.append(key)
            continue
        if value.shape == current[key].shape:
            adapted[key] = value
            loaded.append(key)
            continue
        if key == "embed.weight" and value.ndim == 2 and current[key].ndim == 2 and value.size(1) == current[key].size(1):
            merged = current[key].clone()
            rows = min(value.size(0), merged.size(0))
            merged[:rows] = value[:rows]
            adapted[key] = merged
            partial.append(f"{key} rows={rows}/{merged.size(0)}")
            continue
        skipped.append(key)

    current.update(adapted)
    model.load_state_dict(current)
    print(f"[INFO] fine-tune checkpoint loaded: {model_path}")
    print(f"[INFO] loaded tensors={len(loaded)} partial={partial or 'none'} skipped={len(skipped)}")
    return True


def build_training_sampler(dataset, fine_tune_mode, new_data_weight, old_data_replay_ratio):
    if not fine_tune_mode or len(dataset) == 0:
        return None, {}

    sample_flags = list(getattr(dataset, "sample_is_new_group", []))
    if not sample_flags:
        return None, {}
    new_count = sum(int(flag) for flag in sample_flags)
    old_count = len(sample_flags) - new_count
    if new_count == 0 or old_count == 0:
        return None, {
            "new_row_count": new_count,
            "old_row_count": old_count,
            "enabled": False,
        }

    replay_ratio = min(max(float(old_data_replay_ratio), 0.0), 1.0)
    new_ratio = max(0.0, 1.0 - replay_ratio)
    old_base = replay_ratio / max(1, old_count)
    new_base = new_ratio / max(1, new_count)
    boost = max(1.0, float(new_data_weight))
    weights = [
        (new_base * boost) if int(flag) else old_base
        for flag in sample_flags
    ]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(sample_flags), replacement=True)
    return sampler, {
        "new_row_count": new_count,
        "old_row_count": old_count,
        "enabled": True,
    }


def configure_torch_runtime(device):
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    configure_torch_runtime(device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    for artifact_path in (args.model_path, args.meta_path, args.vocab_path, args.split_manifest_path):
        artifact_dir = os.path.dirname(artifact_path)
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)

    can_resume = args.fine_tune and os.path.exists(args.model_path) and os.path.exists(args.vocab_path)
    log_progress(f"Preprocess start: data={args.data_path} fine_tune={can_resume}")

    log_progress("Step 1/8: load samples")
    samples = load_samples(args.data_path)
    log_progress(f"Loaded samples: {len(samples)}")

    log_progress("Step 2/8: bootstrap normalize samples")
    bootstrap_resolver = NameResolver()
    samples = [
        normalize_relation_sample(sample, bootstrap_resolver)
        for sample in tqdm(samples, desc="bootstrap normalize", unit="sample")
    ]

    log_progress("Step 3/8: build name resources")
    name_resources = build_name_resolution_resources(samples)
    alias_map = name_resources["name_slug_aliases"]
    resolver = NameResolver(alias_map=alias_map, display_name_map=name_resources["display_name_map"])
    samples = [
        normalize_relation_sample(sample, resolver)
        for sample in tqdm(samples, desc="resolver normalize", unit="sample")
    ]

    log_progress("Step 4/8: split train/val and detect changed groups")
    train_samples, _, split_info = split_train_val_incremental(
        samples=samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        manifest_path=args.split_manifest_path,
        rebuild=args.rebuild_split,
    )
    split_manifest = load_split_manifest(args.split_manifest_path)
    grouped_samples = build_grouped_samples(samples)
    change_plan = build_feature_cache_change_plan(grouped_samples, split_manifest, args, can_resume)
    log_progress(
        "Changed group plan: "
        f"changed={change_plan['changed']} changed_groups={len(change_plan['changed_group_keys'])} "
        f"stale_groups={len(change_plan['stale_group_keys'])} forced_rebuild={change_plan['forced_rebuild']}"
    )

    log_progress("Step 5/8: build or extend vocab incrementally")
    vocab = build_or_extend_vocab(
        samples,
        alias_map=alias_map,
        existing_vocab_path=args.vocab_path,
        fine_tune=can_resume,
        grouped_samples=grouped_samples,
        changed_group_keys=change_plan["changed_group_keys"],
    )

    log_progress("Step 6/8: sync SQLite feature cache")
    feature_sync = sync_relation_feature_cache(
        grouped_samples,
        split_manifest,
        vocab,
        alias_map,
        args,
        can_resume,
        change_plan=change_plan,
    )
    log_progress(
        "Feature cache sync: "
        f"changed={feature_sync['changed']} updated_groups={feature_sync['updated_groups']} "
        f"deleted_groups={feature_sync['deleted_groups']} forced_rebuild={feature_sync['forced_rebuild']}"
    )

    cached_payload = None
    if not feature_sync["changed"]:
        cached_payload = load_preprocessed_cache(args, can_resume)
    if cached_payload is not None:
        log_progress(f"Loaded preprocessed cache: {args.preprocessed_cache_path}")
        name_resources = cached_payload["name_resources"]
        alias_map = name_resources["name_slug_aliases"]
        vocab = MonsterVocab(cached_payload["vocab_dict"])
        split_info = cached_payload["split_info"]
        train_pos = cached_payload["train_pos"]
        val_pos = cached_payload["val_pos"]
        train_ban_pos = cached_payload["train_ban_pos"]
        val_ban_pos = cached_payload["val_ban_pos"]
        monster_freq = Counter(cached_payload["monster_freq"])
        pair_freq = Counter(cached_payload["pair_freq"])
        ban_freq = Counter(cached_payload["ban_freq"])
    else:
        log_progress("Step 7/8: load positive actions and frequency stats from SQLite feature cache")
        if args.fine_tune_new_only:
            if not any(bool(sample.get("_is_new_group", False)) for sample in train_samples):
                log_progress("Fine-tune new-only requested but no new train groups were found. Exiting.")
                return
        feature_rows = load_relation_features_from_cache(args.feature_cache_path, train_new_only=args.fine_tune_new_only)
        train_pos = feature_rows["train_pos"]
        val_pos = feature_rows["val_pos"]
        train_ban_pos = feature_rows["train_ban_pos"]
        val_ban_pos = feature_rows["val_ban_pos"]
        monster_freq = Counter(feature_rows["monster_freq"])
        pair_freq = Counter(feature_rows["pair_freq"])
        ban_freq = Counter(int(row["candidate"]) for row in train_ban_pos)

        log_progress("Step 8/8: save preprocessed cache")

        save_preprocessed_cache(
            args,
            can_resume,
            {
                "name_resources": name_resources,
                "vocab_dict": vocab.to_dict(),
                "split_info": split_info,
                "train_pos": train_pos,
                "val_pos": val_pos,
                "train_ban_pos": train_ban_pos,
                "val_ban_pos": val_ban_pos,
                "monster_freq": dict(monster_freq),
                "pair_freq": dict(pair_freq),
                "ban_freq": dict(ban_freq),
            },
        )

    print(f"split manifest {'created' if split_info['manifest_created'] else 'loaded'}: {args.split_manifest_path}")
    print(f"old groups kept: {split_info['old_groups_kept']}")
    print(f"new groups assigned: {split_info['new_groups_assigned']}")
    print(f"train groups: {split_info['train_group_count']}")
    print(f"val groups: {split_info['val_group_count']}")
    print(f"train samples: {split_info['train_sample_count']}")
    print(f"val samples: {split_info['val_sample_count']}")

    print(
        f"relation positives: pick_train={len(train_pos)} pick_val={len(val_pos)} "
        f"ban_train={len(train_ban_pos)} ban_val={len(val_ban_pos)} vocab={len(vocab)} fine_tune={can_resume}"
    )
    print(f"Fine-tune mode: {can_resume}")
    print(f"New train groups: {split_info['new_train_group_count']}")
    print(f"Old train groups: {split_info['old_train_group_count']}")
    print(f"New train samples: {split_info['new_train_sample_count']}")
    print(f"Old train samples: {split_info['old_train_sample_count']}")
    print(f"New data weight: {float(args.new_data_weight):.4f}")
    print(f"Old data replay ratio: {float(args.old_data_replay_ratio):.4f}")
    if can_resume and split_info["new_train_group_count"] == 0:
        print("No new train groups found; using the stable train split without new-data upweighting.")

    train_ds = RelationPairwiseDataset(train_pos, vocab, monster_freq, pair_freq, args.negatives_per_positive, seed=args.seed)
    val_ds = RelationPairwiseDataset(val_pos, vocab, monster_freq, pair_freq, args.negatives_per_positive, seed=args.seed + 1)
    train_ban_ds = BanPairwiseDataset(train_ban_pos, vocab, ban_freq, args.negatives_per_positive, seed=args.seed + 2)
    val_ban_ds = BanPairwiseDataset(val_ban_pos, vocab, ban_freq, args.negatives_per_positive, seed=args.seed + 3)

    pick_sampler, pick_sampler_info = build_training_sampler(
        train_ds,
        fine_tune_mode=can_resume and not args.fine_tune_new_only,
        new_data_weight=args.new_data_weight,
        old_data_replay_ratio=args.old_data_replay_ratio,
    )
    ban_sampler, ban_sampler_info = build_training_sampler(
        train_ban_ds,
        fine_tune_mode=can_resume and not args.fine_tune_new_only,
        new_data_weight=args.new_data_weight,
        old_data_replay_ratio=args.old_data_replay_ratio,
    )

    pin_memory = device_type == "cuda" and args.pin_memory
    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        sampler=pick_sampler,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0 and args.persistent_workers and args.hard_negative_ratio <= 0.0,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0 and args.persistent_workers and args.hard_negative_ratio <= 0.0,
    )
    train_ban_loader = make_loader(
        train_ban_ds,
        batch_size=args.batch_size,
        sampler=ban_sampler,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0 and args.persistent_workers and args.hard_negative_ratio <= 0.0,
    )
    val_ban_loader = make_loader(
        val_ban_ds,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0 and args.persistent_workers and args.hard_negative_ratio <= 0.0,
    )

    model = BPDraftRelationNetV2(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=vocab.get_id(PAD_TOKEN),
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
    ).to(device)
    if can_resume:
        load_relation_checkpoint_for_finetune(model, args.model_path)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")
            args.compile = False

    lr = args.lr if args.lr is not None else (FINETUNE_LR if can_resume else LR)
    epochs = args.epochs if args.epochs is not None else (FINETUNE_EPOCHS if can_resume else EPOCHS)
    optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    use_amp = bool(args.amp and device_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    if device_type == "cuda":
        device_index = torch.device(device).index or 0
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    print(f"AMP enabled: {use_amp}")
    print(f"torch.compile enabled: {bool(args.compile)}")
    print(f"batch_size: {int(args.batch_size)}")
    print(f"grad_accum_steps: {int(args.grad_accum_steps)}")
    print(f"effective_batch_size: {int(args.batch_size) * max(1, int(args.grad_accum_steps))}")
    print(f"num_workers: {int(args.num_workers)}")
    print(f"pin_memory: {bool(pin_memory)}")
    print(
        "persistent_workers: "
        f"{bool(args.num_workers > 0 and args.persistent_workers and args.hard_negative_ratio <= 0.0)}"
    )
    print(f"prefetch_factor: {args.prefetch_factor if args.num_workers > 0 else None}")
    print(f"train dataset size: {len(train_ds) + len(train_ban_ds)}")
    print(f"val dataset size: {len(val_ds) + len(val_ban_ds)}")
    if pick_sampler_info:
        print(f"pick sampler enabled: {pick_sampler_info.get('enabled', False)}")
    if ban_sampler_info:
        print(f"ban sampler enabled: {ban_sampler_info.get('enabled', False)}")

    best_state = None
    best_val = float("inf")
    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        negative_refresh_start = time.perf_counter()
        if hasattr(train_ds, "refresh_negative_pairs"):
            train_ds.refresh_negative_pairs(
                args.seed + epoch,
                model=unwrap_model(model),
                device=device,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_candidate_pool=args.hard_negative_candidate_pool,
            )
        if hasattr(train_ban_ds, "refresh_negative_pairs"):
            train_ban_ds.refresh_negative_pairs(args.seed + 10000 + epoch)
        negative_refresh_time = time.perf_counter() - negative_refresh_start

        train_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            outcome_loss_weight=args.outcome_loss_weight,
            desc_prefix="pick ",
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
        )
        train_ban_metrics = run_epoch(
            model,
            train_ban_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            loss_scale=float(args.ban_loss_weight),
            outcome_loss_weight=args.outcome_loss_weight,
            desc_prefix="ban ",
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
        )
        train_time = time.perf_counter() - train_start
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device=device, desc_prefix="pick ", log_interval=args.log_interval)
            val_ban_metrics = run_epoch(model, val_ban_loader, device=device, desc_prefix="ban ", log_interval=args.log_interval)
            ban_topk_metrics = evaluate_ban_topk(model, val_ban_pos, vocab, device=device)
            pick_topk_metrics = evaluate_pick_topk(
                model,
                val_pos,
                vocab,
                device=device,
                candidate_cap=args.pick_eval_candidate_cap,
                max_samples=args.pick_eval_max_samples,
            )

        combined_val = val_metrics["loss"] + float(args.ban_loss_weight) * val_ban_metrics["loss"]
        epoch_time = time.perf_counter() - epoch_start
        train_samples = len(train_ds) + len(train_ban_ds)
        train_samples_per_sec = train_samples / max(train_time, 1e-6)
        peak_memory_mb = 0.0
        if device_type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        print(
            f"epoch {epoch:02d} "
            f"pick train loss={train_metrics['loss']:.4f} pick val loss={val_metrics['loss']:.4f} "
            f"pick pairwise accuracy={val_metrics['overall_acc']:.4f} "
            f"ban train loss={train_ban_metrics['loss']:.4f} ban val loss={val_ban_metrics['loss']:.4f} "
            f"ban pairwise accuracy={val_ban_metrics['overall_acc']:.4f} "
            f"ban top-1 accuracy={ban_topk_metrics['top1_acc']:.4f} "
            f"ban top-3 accuracy={ban_topk_metrics['top3_acc']:.4f} "
            f"single-action pairwise accuracy={val_metrics['single_acc']:.4f} "
            f"pair-action pairwise accuracy={val_metrics['pair_acc']:.4f} "
            f"single top-1={pick_topk_metrics['single_top1_acc']:.4f} "
            f"single top-3={pick_topk_metrics['single_top3_acc']:.4f} "
            f"pair top-1={pick_topk_metrics['pair_top1_acc']:.4f} "
            f"pair top-3={pick_topk_metrics['pair_top3_acc']:.4f}"
        )
        print(
            f"epoch {epoch:02d} negative refresh time={negative_refresh_time:.2f}s "
            f"train samples/sec={train_samples_per_sec:.2f} "
            f"epoch time={epoch_time:.2f}s "
            f"peak_gpu_memory_mb={peak_memory_mb:.2f}"
        )
        if combined_val < best_val:
            best_val = combined_val
            best_state = {key: value.detach().cpu().clone() for key, value in unwrap_model(model).state_dict().items()}

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    score_calibration = fit_relation_score_calibration(
        unwrap_model(model),
        val_loader,
        device=device,
    )
    torch.save(unwrap_model(model).state_dict(), args.model_path)
    save_vocab(vocab, args.vocab_path)
    meta = {
        "model_type": "bp_draft_relation_net_v2",
        "architecture_version": RELATION_ARCHITECTURE_VERSION,
        "architecture": "chronological_sequence_transformer_with_action_state_cross_attention",
        "data_path": args.data_path,
        "training_mode": "fine_tune" if can_resume else "from_scratch",
        "epochs": int(epochs),
        "lr": float(lr),
        "embed_dim": EMBED_DIM,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 512,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "vocab_size": len(vocab),
        "negatives_per_positive": args.negatives_per_positive,
        "ban_loss_weight": float(args.ban_loss_weight),
        "outcome_loss_weight": float(args.outcome_loss_weight),
        "has_ban_head": True,
        "action_types": {
            "single_pick": 0,
            "pair_pick": 1,
            "ban": 2,
        },
        "loss": "pairwise_softplus_with_inverse_propensity_frequency_debias",
        "hard_negative_ratio": float(args.hard_negative_ratio),
        "hard_negative_candidate_pool": int(args.hard_negative_candidate_pool),
        "pick_eval_candidate_cap": int(args.pick_eval_candidate_cap),
        "pick_eval_max_samples": int(args.pick_eval_max_samples),
        "relation_score_center": score_calibration["center"],
        "relation_score_scale": score_calibration["scale"],
        "name_slug_aliases": alias_map,
        "display_name_map": name_resources["display_name_map"],
        "image_name_map": name_resources["image_name_map"],
        "monster_frequency": {str(k): int(v) for k, v in monster_freq.items()},
        "pair_frequency": {f"{a},{b}": int(v) for (a, b), v in pair_freq.items()},
        "ban_frequency": {str(k): int(v) for k, v in ban_freq.items()},
        "split_manifest_path": args.split_manifest_path,
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "train_group_count": int(split_info["train_group_count"]),
        "val_group_count": int(split_info["val_group_count"]),
        "notes": [
            "Scores state/action compatibility directly.",
            "Single actions use candidate_2=PAD and action_mask=[1,0].",
            "Pair actions are scored by the relation model as unordered pair actions.",
            "Ban actions use candidate_1=enemy candidate, candidate_2=PAD, action_mask=[1,0], action_type=2.",
            "Negative sampling uses balanced high-frequency, near-frequency, random, and model-mined candidates.",
            "Frequency correction is bounded inverse-propensity weighting rather than a positive frequency boost.",
            "Pick positives include both player perspectives and preserve the chronological interleaving inside the state encoder.",
            "Observed match outcome is an auxiliary action-value target for pick and ban positives.",
            "Top-k pick metrics evaluate the candidate set instead of only sampled pairwise negatives.",
            "Stable split manifest preserves prior group assignments across incremental fine-tunes.",
        ],
    }
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved model: {args.model_path}")
    print(f"Saved meta : {args.meta_path}")
    print(f"Saved vocab: {args.vocab_path}")


if __name__ == "__main__":
    main()
