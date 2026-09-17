import argparse
import json
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bp_data_common import (
    NameResolver,
    build_name_resolution_resources,
    load_record_rows,
    normalize_relation_sample,
)
from bp_prefix_value_common import (
    BPPrefixValueNet,
    build_prefix_value_vocab,
    encode_prefix_list,
)
from bp_transformer_common import PAD_TOKEN, PROJECT_ROOT, normalize_name
from rank_tiers import get_rank_context_from_record

DATA_PATH = str(PROJECT_ROOT / "data" / "bp_samples.sqlite")
SAVE_DIR = str(PROJECT_ROOT / "models")
MODEL_PATH = os.path.join(SAVE_DIR, "bp_prefix_value_net.pt")
META_PATH = os.path.join(SAVE_DIR, "bp_prefix_value_meta.json")
VOCAB_PATH = os.path.join(SAVE_DIR, "bp_prefix_value_vocab.json")

BATCH_SIZE = 256
EPOCHS = 10
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
TEMPERATURE_CANDIDATES = [1.0 + 0.25 * i for i in range(0, 21)]


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

    value = sample.get("my_result")
    if isinstance(value, str):
        value = normalize_name(value)
        if value in {"win", "victory", "winner"}:
            return 1.0
        if value in {"lose", "loss", "defeat"}:
            return 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0

    if "result" in sample:
        return None

    value = sample.get("won")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
    return None


def _sample_slugs(sample, side, resolver):
    slug_key = f"{side}_picks_slugs"
    name_key = f"{side}_picks"
    values = sample.get(slug_key) or sample.get(name_key) or []
    return [resolver.to_slug(value) for value in values if resolver.to_slug(value)]


def _leader_slug(sample, side, resolver):
    slug = sample.get(f"{side}_leader_slug")
    value = slug or sample.get(f"{side}_leader")
    return resolver.to_slug(value) if value else ""


def normalize_prefix_samples(samples):
    resources = build_name_resolution_resources(samples)
    resolver = NameResolver(
        alias_map=resources["name_slug_aliases"],
        display_name_map=resources["display_name_map"],
    )
    normalized = []
    for sample in samples:
        item = normalize_relation_sample(sample, resolver)
        label = parse_result_label(item)
        if label is None:
            continue
        my_picks = _sample_slugs(item, "my", resolver)
        enemy_picks = _sample_slugs(item, "enemy", resolver)
        first_pick_side = normalize_name(item.get("first_pick_side"))
        current_side = normalize_name(item.get("current_side"))
        if first_pick_side not in {"me", "enemy"} or current_side not in {"me", "enemy"}:
            continue
        normalized.append(
            {
                "match_id": str(item.get("match_id") or item.get("sample_id") or ""),
                "my_picks": my_picks[:5],
                "enemy_picks": enemy_picks[:5],
                "my_leader": _leader_slug(item, "my", resolver),
                "enemy_leader": _leader_slug(item, "enemy", resolver),
                "turn": int(item.get("turn", len(my_picks) + len(enemy_picks) + 1) or 1),
                "first_pick_side": 1.0 if first_pick_side == "me" else 0.0,
                "current_side": 1.0 if current_side == "me" else 0.0,
                "label": float(label),
                "sample_weight": float(get_rank_context_from_record(item)["sample_weight"]),
            }
        )
    return normalized, resources


def split_by_match(samples, val_ratio, seed):
    groups = {}
    for sample in samples:
        groups.setdefault(sample["match_id"], []).append(sample)
    match_ids = list(groups)
    random.Random(seed).shuffle(match_ids)
    val_count = max(1, int(len(match_ids) * float(val_ratio)))
    val_ids = set(match_ids[:val_count])
    train_samples = []
    val_samples = []
    for match_id, rows in groups.items():
        (val_samples if match_id in val_ids else train_samples).extend(rows)
    return train_samples, val_samples


class PrefixValueDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = list(samples)
        self.vocab = vocab
        self.pad_id = vocab.get_id(PAD_TOKEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        my_ids, my_mask = encode_prefix_list(sample["my_picks"], self.vocab)
        enemy_ids, enemy_mask = encode_prefix_list(sample["enemy_picks"], self.vocab)
        my_leader = self.vocab.get_id(sample["my_leader"]) if sample["my_leader"] else self.pad_id
        enemy_leader = self.vocab.get_id(sample["enemy_leader"]) if sample["enemy_leader"] else self.pad_id
        return {
            "my_picks": torch.tensor(my_ids, dtype=torch.long),
            "my_mask": torch.tensor(my_mask, dtype=torch.float32),
            "enemy_picks": torch.tensor(enemy_ids, dtype=torch.long),
            "enemy_mask": torch.tensor(enemy_mask, dtype=torch.float32),
            "my_ban": torch.tensor(self.pad_id, dtype=torch.long),
            "enemy_ban": torch.tensor(self.pad_id, dtype=torch.long),
            "my_leader": torch.tensor(my_leader, dtype=torch.long),
            "enemy_leader": torch.tensor(enemy_leader, dtype=torch.long),
            "turn": torch.tensor(float(sample["turn"]), dtype=torch.float32),
            "first_pick_side": torch.tensor(float(sample["first_pick_side"]), dtype=torch.float32),
            "current_side": torch.tensor(float(sample["current_side"]), dtype=torch.float32),
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
        }


def collate_fn(batch):
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in batch[0]}


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_forward(model, batch):
    return model(
        my_picks=batch["my_picks"],
        my_mask=batch["my_mask"],
        enemy_picks=batch["enemy_picks"],
        enemy_mask=batch["enemy_mask"],
        my_ban=batch["my_ban"],
        enemy_ban=batch["enemy_ban"],
        my_leader=batch["my_leader"],
        enemy_leader=batch["enemy_leader"],
        turn=batch["turn"],
        first_pick_side=batch["first_pick_side"],
        current_side=batch["current_side"],
    )


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_brier = 0.0
    total_accuracy = 0.0
    total_weight = 0.0

    for batch in tqdm(loader, leave=False):
        batch = move_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model_forward(model, batch)
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["label"],
                reduction="none",
            )
            weight_sum = batch["sample_weight"].sum().clamp_min(1e-6)
            loss = (per_sample_loss * batch["sample_weight"]).sum() / weight_sum
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        probs = torch.sigmoid(logits)
        hard_target = (batch["label"] >= 0.5).float()
        pred = (probs >= 0.5).float()
        batch_weight = float(batch["sample_weight"].sum().item())
        brier = (((probs - batch["label"]) ** 2) * batch["sample_weight"]).sum() / weight_sum
        accuracy = (((pred == hard_target).float()) * batch["sample_weight"]).sum() / weight_sum
        total_loss += float(loss.item()) * batch_weight
        total_brier += float(brier.item()) * batch_weight
        total_accuracy += float(accuracy.item()) * batch_weight
        total_weight += batch_weight

    return {
        "loss": total_loss / max(total_weight, 1e-6),
        "brier": total_brier / max(total_weight, 1e-6),
        "hard_acc": total_accuracy / max(total_weight, 1e-6),
    }


@torch.no_grad()
def collect_eval_outputs(model, loader, device):
    model.eval()
    logits = []
    labels = []
    weights = []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits.append(model_forward(model, batch).cpu())
        labels.append(batch["label"].cpu())
        weights.append(batch["sample_weight"].cpu())
    if not logits:
        empty = torch.empty(0)
        return empty, empty, empty
    return torch.cat(logits), torch.cat(labels), torch.cat(weights)


def weighted_bce(logits, labels, weights, temperature=1.0):
    scaled = logits / max(float(temperature), 1e-6)
    loss = F.binary_cross_entropy_with_logits(scaled, labels, reduction="none")
    return float((loss * weights).sum().div(weights.sum().clamp_min(1e-6)).item())


def find_best_temperature(logits, labels, weights):
    best_temperature = 1.0
    best_loss = weighted_bce(logits, labels, weights, best_temperature)
    for temperature in TEMPERATURE_CANDIDATES:
        loss = weighted_bce(logits, labels, weights, temperature)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature, best_loss


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chronological BP prefix value model.")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--meta-path", default=META_PATH)
    parser.add_argument("--vocab-path", default=VOCAB_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    raw_samples = load_record_rows(args.data_path, "bp_samples")
    print(f"loaded prefix samples: {len(raw_samples)}")
    samples, resources = normalize_prefix_samples(raw_samples)
    print(f"usable prefix samples: {len(samples)}")
    if not samples:
        raise ValueError(f"{args.data_path} 中没有可用的逐回合胜负样本。")

    train_samples, val_samples = split_by_match(samples, args.val_ratio, args.seed)
    vocab = build_prefix_value_vocab(train_samples)
    train_ds = PrefixValueDataset(train_samples, vocab)
    val_ds = PrefixValueDataset(val_samples, vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BPPrefixValueNet(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=vocab.get_id(PAD_TOKEN),
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        print(
            f"epoch {epoch:02d} "
            f"train loss={train_metrics['loss']:.4f} brier={train_metrics['brier']:.4f} "
            f"val loss={val_metrics['loss']:.4f} brier={val_metrics['brier']:.4f} "
            f"val acc={val_metrics['hard_acc']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits, val_labels, val_weights = collect_eval_outputs(model, val_loader, device)
    best_temperature, calibrated_val_loss = find_best_temperature(val_logits, val_labels, val_weights)
    uncalibrated_val_loss = weighted_bce(val_logits, val_labels, val_weights, 1.0)

    torch.save(model.state_dict(), args.model_path)
    save_vocab(vocab, args.vocab_path)
    meta = {
        "model_type": "bp_prefix_value_transformer_v1",
        "architecture": "transformer_encoder",
        "data_path": args.data_path,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ffn_dim": FFN_DIM,
        "numeric_features": ["turn", "first_pick_side", "current_side"],
        "ban_features": "masked_during_prefix_training",
        "pad_token": PAD_TOKEN,
        "vocab_size": len(vocab),
        "device_trained": device,
        "label_mode": "match_result",
        "raw_sample_count": len(samples),
        "train_sample_count": len(train_ds),
        "val_sample_count": len(val_ds),
        "prob_temperature": best_temperature,
        "prob_prior_weight": 0.0,
        "prob_clip_min": 1e-4,
        "prob_clip_max": 1.0 - 1e-4,
        "uncalibrated_val_loss": uncalibrated_val_loss,
        "calibrated_val_loss": calibrated_val_loss,
        "best_val_loss": best_val_loss,
        "name_slug_aliases": resources["name_slug_aliases"],
        "display_name_map": resources["display_name_map"],
        "image_name_map": resources["image_name_map"],
        "notes": [
            "uses chronological prefixes from bp_samples rather than full-draft leave-one-out states",
            "groups train and validation by match_id",
            "final ban fields are masked to avoid future leakage during pick evaluation",
            "output probability is calibrated on the held-out prefix split",
        ],
    }
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved model: {args.model_path}")
    print(f"saved meta : {args.meta_path}")
    print(f"saved vocab: {args.vocab_path}")


if __name__ == "__main__":
    main()
