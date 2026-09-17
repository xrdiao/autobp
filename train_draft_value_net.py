import json
import math
import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bp_data_common import load_record_rows
from bp_transformer_common import (
    PAD_TOKEN,
    PROJECT_ROOT,
    SequenceFeatureTransformer,
    UNK_TOKEN,
    normalize_name,
    slugify_fallback,
)
from dataset_versions import summarize_dataset_version_counts
from rank_tiers import RANK_TIER_WEIGHTS, get_rank_context_from_record, summarize_rank_tier_counts

DATA_PATH = str(PROJECT_ROOT / "data" / "bp_samples.sqlite")
SAVE_DIR = str(PROJECT_ROOT / "models")

MODEL_PATH = os.path.join(SAVE_DIR, "bp_draft_value_net.pt")
META_PATH = os.path.join(SAVE_DIR, "bp_draft_value_meta.json")
VOCAB_PATH = os.path.join(SAVE_DIR, "bp_draft_value_vocab.json")

BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3
FFN_DIM = 384
DROPOUT = 0.15
VAL_RATIO = 0.1
SEED = 42
SAMPLE_WEIGHT_MODE = "sqrt_count"
TEMPERATURE_CANDIDATES = [1.0 + 0.25 * i for i in range(0, 21)]
PROB_PRIOR_WEIGHT = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_result_label(sample):
    if "result" in sample:
        value = normalize_name(sample["result"])
        if value in ["win", "lose", "loss"]:
            return 1.0 if value == "win" else 0.0

    if "my_result" in sample:
        value = sample["my_result"]
        if isinstance(value, str):
            value = normalize_name(value)
            if value in ["win", "lose", "loss"]:
                return 1.0 if value == "win" else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if float(value) > 0 else 0.0

    if "is_win" in sample:
        return 1.0 if bool(sample["is_win"]) else 0.0

    if "won" in sample:
        value = sample["won"]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if float(value) > 0 else 0.0

    if "winner" in sample:
        value = normalize_name(sample["winner"])
        if value in ["me", "self", "player", "win"]:
            return 1.0
        if value in ["enemy", "opponent", "lose", "loss"]:
            return 0.0

    return None


def get_sample_rank_weight(sample):
    return float(get_rank_context_from_record(sample)["sample_weight"])


class MonsterVocab:
    def __init__(self, slug_to_id=None):
        if slug_to_id is None:
            slug_to_id = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
            }
        self.slug_to_id = dict(slug_to_id)
        self.id_to_slug = {v: k for k, v in self.slug_to_id.items()}

    def __len__(self):
        return len(self.slug_to_id)

    def add_token(self, slug):
        slug = normalize_name(slug)
        if slug and slug not in self.slug_to_id:
            idx = len(self.slug_to_id)
            self.slug_to_id[slug] = idx
            self.id_to_slug[idx] = slug

    def get_id(self, slug):
        slug = normalize_name(slug)
        if not slug:
            return self.slug_to_id[UNK_TOKEN]
        return self.slug_to_id.get(slug, self.slug_to_id[UNK_TOKEN])

    def to_dict(self):
        return dict(self.slug_to_id)


def _first_list(sample, keys):
    for key in keys:
        values = sample.get(key)
        if values:
            return [slugify_fallback(value) for value in values if value][:5]
    return []


def _pick_slugs(sample, side):
    return _first_list(
        sample,
        (
            f"final_{side}_pick_slugs",
            f"{side}_pick_slugs",
            f"{side}_picks_slugs",
            f"final_{side}_picks",
            f"{side}_picks",
        ),
    )


def _ban_slug(sample, side):
    for key in (f"{side}_ban_slug", f"{side}_ban"):
        value = sample.get(key)
        if value:
            return slugify_fallback(value)
    return ""


def _leader_slug(sample, side):
    for key in (f"{side}_leader_slug", f"{side}_leader"):
        value = sample.get(key)
        if value:
            return slugify_fallback(value)
    return ""


def normalize_full_draft_sample(sample):
    item = dict(sample)
    item["my_pick_slugs"] = _pick_slugs(item, "my")
    item["enemy_pick_slugs"] = _pick_slugs(item, "enemy")
    item["my_ban"] = _ban_slug(item, "my")
    item["enemy_ban"] = _ban_slug(item, "enemy")
    item["my_leader"] = _leader_slug(item, "my")
    item["enemy_leader"] = _leader_slug(item, "enemy")
    return item


def extract_slugs_from_full_sample(sample):
    normalized = normalize_full_draft_sample(sample)
    out = []

    for key in ["my_pick_slugs", "enemy_pick_slugs"]:
        for monster in normalized.get(key, []) or []:
            out.append(normalize_name(monster))

    for key in ["my_ban", "enemy_ban", "my_leader", "enemy_leader"]:
        value = normalized.get(key)
        if value:
            out.append(normalize_name(value))

    return [x for x in out if x]


def build_full_draft_signature(sample):
    normalized = normalize_full_draft_sample(sample)
    return (
        tuple(normalized["my_pick_slugs"]),
        tuple(normalized["enemy_pick_slugs"]),
        normalized["my_ban"],
        normalized["enemy_ban"],
        normalized["my_leader"],
        normalized["enemy_leader"],
    )


def build_aggregated_sample(first_sample, labels, rank_weights, weight_mode="sqrt_count"):
    support_count = len(labels)
    if support_count <= 0:
        return None

    weight_sum = float(sum(rank_weights)) if rank_weights else 0.0
    if weight_sum > 0:
        soft_label = float(sum(label * weight for label, weight in zip(labels, rank_weights)) / weight_sum)
    else:
        soft_label = float(sum(labels) / support_count)

    mean_rank_weight = (weight_sum / support_count) if support_count > 0 else 1.0
    if weight_mode == "count":
        sample_weight = weight_sum if weight_sum > 0 else float(support_count)
    elif weight_mode == "sqrt_count":
        sample_weight = float(math.sqrt(support_count)) * mean_rank_weight
    else:
        sample_weight = mean_rank_weight

    normalized = normalize_full_draft_sample(first_sample)
    rank_context = get_rank_context_from_record(normalized)
    return {
        "my_pick_slugs": normalized["my_pick_slugs"],
        "enemy_pick_slugs": normalized["enemy_pick_slugs"],
        "my_ban": normalized["my_ban"],
        "enemy_ban": normalized["enemy_ban"],
        "my_leader": normalized["my_leader"],
        "enemy_leader": normalized["enemy_leader"],
        "rank_tier": rank_context["rank_tier"],
        "rank_tier_label": rank_context["rank_tier_label"],
        "soft_label": soft_label,
        "support_count": int(support_count),
        "win_count": float(sum(labels)),
        "loss_count": float(support_count - sum(labels)),
        "sample_weight": float(sample_weight),
        "rank_weight_mean": float(mean_rank_weight),
        "rank_weight_sum": float(weight_sum),
    }


def aggregate_samples_by_signature(samples, weight_mode="sqrt_count"):
    groups = {}
    for sample in samples:
        label = parse_result_label(sample)
        if label is None:
            continue
        signature = build_full_draft_signature(sample)
        bucket = groups.setdefault(signature, {"first": sample, "labels": [], "rank_weights": []})
        bucket["labels"].append(float(label))
        bucket["rank_weights"].append(get_sample_rank_weight(sample))

    aggregated = []
    for bucket in groups.values():
        item = build_aggregated_sample(
            first_sample=bucket["first"],
            labels=bucket["labels"],
            rank_weights=bucket["rank_weights"],
            weight_mode=weight_mode,
        )
        if item is not None:
            aggregated.append(item)
    return aggregated


def summarize_aggregated_samples(samples):
    supports = [int(s.get("support_count", 1)) for s in samples]
    soft_labels = [float(s.get("soft_label", 0.0)) for s in samples]
    mixed_states = sum(1 for value in soft_labels if 0.0 < value < 1.0)
    return {
        "aggregated_state_count": len(samples),
        "mixed_soft_label_states": mixed_states,
        "support_mean": float(sum(supports) / max(1, len(supports))),
        "support_max": max(supports) if supports else 0,
        "soft_label_mean": float(sum(soft_labels) / max(1, len(soft_labels))),
    }


def build_vocab(samples):
    vocab = MonsterVocab()
    for sample in samples:
        for slug in extract_slugs_from_full_sample(sample):
            vocab.add_token(slug)
    return vocab


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)


class DraftValueDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = []
        self.vocab = vocab

        for sample in samples:
            label = sample.get("soft_label")
            if label is None:
                label = parse_result_label(sample)
            if label is None:
                continue

            self.samples.append(
                {
                    "my_picks": [normalize_name(x) for x in (sample.get("my_pick_slugs") or []) if x][:5],
                    "enemy_picks": [normalize_name(x) for x in (sample.get("enemy_pick_slugs") or []) if x][:5],
                    "my_ban": slugify_fallback(sample.get("my_ban")) if sample.get("my_ban") else "",
                    "enemy_ban": slugify_fallback(sample.get("enemy_ban")) if sample.get("enemy_ban") else "",
                    "my_leader": slugify_fallback(sample.get("my_leader")) if sample.get("my_leader") else "",
                    "enemy_leader": slugify_fallback(sample.get("enemy_leader")) if sample.get("enemy_leader") else "",
                    "label": float(label),
                    "sample_weight": float(sample.get("sample_weight", 1.0)),
                    "support_count": float(sample.get("support_count", 1.0)),
                }
            )

    def __len__(self):
        return len(self.samples)

    def encode_list(self, arr, max_len=5):
        ids = [self.vocab.get_id(x) for x in arr[:max_len]]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.vocab.get_id(PAD_TOKEN))
            mask.append(0)
        return ids, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        my_ids, my_mask = self.encode_list(sample["my_picks"], 5)
        enemy_ids, enemy_mask = self.encode_list(sample["enemy_picks"], 5)
        my_ban_id = self.vocab.get_id(sample["my_ban"]) if sample["my_ban"] else self.vocab.get_id(PAD_TOKEN)
        enemy_ban_id = self.vocab.get_id(sample["enemy_ban"]) if sample["enemy_ban"] else self.vocab.get_id(PAD_TOKEN)
        my_leader_id = self.vocab.get_id(sample["my_leader"]) if sample["my_leader"] else self.vocab.get_id(PAD_TOKEN)
        enemy_leader_id = self.vocab.get_id(sample["enemy_leader"]) if sample["enemy_leader"] else self.vocab.get_id(PAD_TOKEN)

        return {
            "my_picks": torch.tensor(my_ids, dtype=torch.long),
            "my_mask": torch.tensor(my_mask, dtype=torch.float32),
            "enemy_picks": torch.tensor(enemy_ids, dtype=torch.long),
            "enemy_mask": torch.tensor(enemy_mask, dtype=torch.float32),
            "my_ban": torch.tensor(my_ban_id, dtype=torch.long),
            "enemy_ban": torch.tensor(enemy_ban_id, dtype=torch.long),
            "my_leader": torch.tensor(my_leader_id, dtype=torch.long),
            "enemy_leader": torch.tensor(enemy_leader_id, dtype=torch.long),
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
            "support_count": torch.tensor(float(sample["support_count"]), dtype=torch.float32),
        }


class BPDraftValueNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        dropout=0.15,
        pad_idx=0,
        num_heads=4,
        num_layers=3,
        ffn_dim=384,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.backbone = SequenceFeatureTransformer(
            vocab_size=vocab_size,
            role_vocab_size=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            pad_idx=pad_idx,
            max_seq_len=15,
            numeric_dim=0,
        )
        self.register_buffer(
            "role_ids",
            torch.tensor([1] * 5 + [2] * 5 + [3] + [4] + [5] + [6], dtype=torch.long),
            persistent=False,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        my_picks,
        my_mask,
        enemy_picks,
        enemy_mask,
        my_ban,
        enemy_ban,
        my_leader,
        enemy_leader,
    ):
        input_ids = torch.cat(
            [
                my_picks,
                enemy_picks,
                my_ban.unsqueeze(1),
                enemy_ban.unsqueeze(1),
                my_leader.unsqueeze(1),
                enemy_leader.unsqueeze(1),
            ],
            dim=1,
        )
        valid_mask = torch.cat(
            [
                my_mask.bool(),
                enemy_mask.bool(),
                (my_ban != self.pad_idx).unsqueeze(1),
                (enemy_ban != self.pad_idx).unsqueeze(1),
                (my_leader != self.pad_idx).unsqueeze(1),
                (enemy_leader != self.pad_idx).unsqueeze(1),
            ],
            dim=1,
        )
        role_ids = self.role_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        _, cls_vec = self.backbone(
            input_ids=input_ids,
            valid_mask=valid_mask,
            role_ids=role_ids,
            numeric_features=None,
        )
        hidden = self.head(cls_vec)
        return self.value_head(hidden).squeeze(-1)


def load_samples(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".sqlite", ".db"}:
        return load_record_rows(path, "bp_samples")
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


def split_train_val(samples, val_ratio=0.1, seed=42):
    groups = {}
    for sample in samples:
        signature = build_full_draft_signature(sample)
        groups.setdefault(signature, []).append(sample)

    signatures = list(groups.keys())
    random.Random(seed).shuffle(signatures)
    n_val = max(1, int(len(signatures) * val_ratio))
    val_signatures = set(signatures[:n_val])

    train_samples = []
    val_samples = []
    for signature, grouped in groups.items():
        if signature in val_signatures:
            val_samples.extend(grouped)
        else:
            train_samples.extend(grouped)
    return train_samples, val_samples


def collate_fn(batch):
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in batch[0]}


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_brier = 0.0
    total_hard_acc = 0.0
    total_weight = 0.0

    for batch in tqdm(loader, leave=False):
        batch = move_to_device(batch, DEVICE)
        with torch.set_grad_enabled(is_train):
            logits = model(
                my_picks=batch["my_picks"],
                my_mask=batch["my_mask"],
                enemy_picks=batch["enemy_picks"],
                enemy_mask=batch["enemy_mask"],
                my_ban=batch["my_ban"],
                enemy_ban=batch["enemy_ban"],
                my_leader=batch["my_leader"],
                enemy_leader=batch["enemy_leader"],
            )

            per_sample_loss = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
            weight_sum = batch["sample_weight"].sum().clamp(min=1e-6)
            loss = (per_sample_loss * batch["sample_weight"]).sum() / weight_sum

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        hard_target = (batch["label"] >= 0.5).float()
        batch_weight = float(batch["sample_weight"].sum().item())

        weighted_brier = (((probs - batch["label"]) ** 2) * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp(min=1e-6)
        weighted_hard_acc = (((pred == hard_target).float()) * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp(min=1e-6)

        total_loss += loss.item() * batch_weight
        total_brier += weighted_brier.item() * batch_weight
        total_hard_acc += weighted_hard_acc.item() * batch_weight
        total_weight += batch_weight

    return {
        "loss": total_loss / max(1e-6, total_weight),
        "brier": total_brier / max(1e-6, total_weight),
        "hard_acc": total_hard_acc / max(1e-6, total_weight),
    }


@torch.no_grad()
def collect_eval_outputs(model, loader):
    model.eval()
    logits_list = []
    labels_list = []
    weights_list = []

    for batch in loader:
        batch = move_to_device(batch, DEVICE)
        logits = model(
            my_picks=batch["my_picks"],
            my_mask=batch["my_mask"],
            enemy_picks=batch["enemy_picks"],
            enemy_mask=batch["enemy_mask"],
            my_ban=batch["my_ban"],
            enemy_ban=batch["enemy_ban"],
            my_leader=batch["my_leader"],
            enemy_leader=batch["enemy_leader"],
        )
        logits_list.append(logits.detach().cpu())
        labels_list.append(batch["label"].detach().cpu())
        weights_list.append(batch["sample_weight"].detach().cpu())

    if not logits_list:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(weights_list, dim=0)


def compute_weighted_bce_from_logits(logits, labels, weights, temperature=1.0):
    if logits.numel() == 0:
        return float("inf")
    scaled_logits = logits / max(float(temperature), 1e-6)
    losses = F.binary_cross_entropy_with_logits(scaled_logits, labels, reduction="none")
    denom = weights.sum().clamp(min=1e-6)
    return float(((losses * weights).sum() / denom).item())


def find_best_temperature(logits, labels, weights, candidates=None):
    if candidates is None:
        candidates = TEMPERATURE_CANDIDATES

    best_temperature = 1.0
    best_loss = compute_weighted_bce_from_logits(logits, labels, weights, temperature=best_temperature)
    for temp in candidates:
        loss = compute_weighted_bce_from_logits(logits, labels, weights, temperature=temp)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temp)
    return best_temperature, best_loss


def main():
    set_seed(SEED)
    ensure_dir(SAVE_DIR)

    samples = load_samples(DATA_PATH)
    print(f"loaded full draft samples: {len(samples)}")
    samples = [normalize_full_draft_sample(sample) for sample in samples]

    filtered = [sample for sample in samples if parse_result_label(sample) is not None]
    print(f"samples with result label: {len(filtered)}")
    print(f"raw dataset version counts: {summarize_dataset_version_counts(filtered)}")
    if not filtered:
        raise ValueError(f"{DATA_PATH} 中没有找到可用胜负字段。")

    aggregated_samples = aggregate_samples_by_signature(filtered, weight_mode=SAMPLE_WEIGHT_MODE)
    agg_stats = summarize_aggregated_samples(aggregated_samples)
    print(f"aggregated states: {agg_stats['aggregated_state_count']}")
    print(
        "aggregated stats:",
        {
            "mixed_soft_label_states": agg_stats["mixed_soft_label_states"],
            "support_mean": round(agg_stats["support_mean"], 4),
            "support_max": agg_stats["support_max"],
            "soft_label_mean": round(agg_stats["soft_label_mean"], 4),
        },
    )

    train_samples, val_samples = split_train_val(aggregated_samples, val_ratio=VAL_RATIO, seed=SEED)
    print(f"train rank tier counts: {summarize_rank_tier_counts(train_samples)}")
    print(f"val rank tier counts: {summarize_rank_tier_counts(val_samples)}")

    vocab = build_vocab(train_samples)
    print(f"vocab size: {len(vocab)}")

    train_ds = DraftValueDataset(train_samples, vocab)
    val_ds = DraftValueDataset(val_samples, vocab)
    print(f"train usable: {len(train_ds)}")
    print(f"val usable: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BPDraftValueNet(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=vocab.get_id(PAD_TOKEN),
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

        print("train:", {k: round(v, 4) for k, v in train_metrics.items()})
        print("val  :", {k: round(v, 4) for k, v in val_metrics.items()})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(model.state_dict())
            print(f"new best val loss: {best_val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_labels, val_weights = collect_eval_outputs(model, val_loader)
    best_temperature, calibrated_val_loss = find_best_temperature(val_logits, val_labels, val_weights, candidates=TEMPERATURE_CANDIDATES)
    uncalibrated_val_loss = compute_weighted_bce_from_logits(val_logits, val_labels, val_weights, temperature=1.0)
    print(
        "temperature calibration:",
        {
            "uncalibrated_val_loss": round(uncalibrated_val_loss, 6),
            "best_temperature": round(best_temperature, 4),
            "calibrated_val_loss": round(calibrated_val_loss, 6),
        },
    )

    torch.save(model.state_dict(), MODEL_PATH)
    save_vocab(vocab, VOCAB_PATH)

    meta = {
        "model_type": "bp_draft_value_transformer_v2",
        "architecture": "transformer_encoder",
        "data_path": DATA_PATH,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ffn_dim": FFN_DIM,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "vocab_size": len(vocab),
        "device_trained": DEVICE,
        "label_mode": "aggregated_soft_label",
        "sample_weight_mode": SAMPLE_WEIGHT_MODE,
        "raw_sample_count": len(filtered),
        "aggregated_state_count": agg_stats["aggregated_state_count"],
        "mixed_soft_label_states": agg_stats["mixed_soft_label_states"],
        "prob_temperature": best_temperature,
        "prob_prior_weight": PROB_PRIOR_WEIGHT,
        "prob_clip_min": 1e-4,
        "prob_clip_max": 1.0 - 1e-4,
        "rank_tier_weights": RANK_TIER_WEIGHTS,
        "raw_dataset_version_counts": summarize_dataset_version_counts(filtered),
        "best_val_loss": float(best_val_loss),
        "uncalibrated_val_loss": float(uncalibrated_val_loss),
        "calibrated_val_loss": float(calibrated_val_loss),
        "notes": [
            "transformer encoder full-draft evaluator for counterfactual ban recommendation",
            "trained from the current chronological bp_samples.sqlite full-draft states",
            "validation split is grouped by full draft signature",
            "sample aggregation and loss both respect rank tier and dataset version weights",
        ],
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved vocab: {VOCAB_PATH}")
    print(f"Saved meta : {META_PATH}")


if __name__ == "__main__":
    main()
