"""Train the shared, event-aware BP network.

This is the sole training entry point for the BP model.  It builds one shared
state encoder and four heads: our pick, opponent pick, ban, and state value.
A pair is represented as one unordered action, while the event sequence used
by the encoder keeps the chronological BP order.

Example::

    python -m training.shared_bp --data-path data/bp_samples.sqlite \
        --epochs 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bp_core.bp_network import BPSharedNetwork
from bp_core.action_space import DraftActionEvent, DraftTimeline, coerce_action_event
from bp_core.data_common import NameResolver, build_name_resolution_resources, normalize_bp_sample
from bp_core.bp_common import MonsterVocab, side_for_order
from bp_core.names import PAD_TOKEN, PROJECT_ROOT
from training.data_pipeline import (
    build_ban_positive_actions,
    build_positive_actions,
    extract_all_slugs_from_sample,
    load_samples,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bp_samples.sqlite"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "models" / "bp_shared"


def _first_side_value(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "me":
            return 1.0
        if value == "enemy":
            return 0.0
        raise ValueError(f"invalid first_pick_side: {value!r}")
    value = float(value)
    if value not in {0.0, 1.0}:
        raise ValueError("first_pick_side must encode me/enemy as 1/0")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the event-aware shared BP network.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--negative-count", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def split_samples(samples, val_ratio: float, seed: int):
    """Split by match/replay group so one draft cannot leak across splits."""

    groups = {}
    for index, sample in enumerate(samples):
        key = str(
            sample.get("_split_group_key")
            or sample.get("match_id")
            or sample.get("replay_id")
            or f"row-{index}"
        )
        groups.setdefault(key, []).append(sample)
    keys = list(groups)
    random.Random(int(seed)).shuffle(keys)
    target = max(1, int(round(len(samples) * max(0.0, min(0.5, float(val_ratio))))))
    val_keys = set()
    count = 0
    for key in keys:
        if count >= target and val_keys:
            break
        val_keys.add(key)
        count += len(groups[key])
    if len(keys) <= 1:
        val_keys.clear()
    for key, rows in groups.items():
        for row in rows:
            row["_split_group_key"] = key
    train = [row for key in keys if key not in val_keys for row in groups[key]]
    valid = [row for key in keys if key in val_keys for row in groups[key]]
    return train, valid


def save_split_manifest(samples, train_samples, valid_samples, path, seed, val_ratio):
    train_groups = {row.get("_split_group_key") for row in train_samples}
    groups = {}
    for row in samples:
        key = row.get("_split_group_key")
        if not key:
            continue
        groups.setdefault(key, {"split": "train" if key in train_groups else "val", "sample_count": 0})
        groups[key]["sample_count"] += 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"version": 1, "seed": int(seed), "val_ratio": float(val_ratio), "groups": groups},
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_vocab(samples):
    vocab = MonsterVocab()
    for sample in samples:
        for slug in extract_all_slugs_from_sample(sample):
            vocab.add_token(slug)
    return vocab


def encode_state(row, vocab, max_events):
    """Encode the canonical chronological event timeline."""

    event_ids = [[0, 0] for _ in range(int(max_events))]
    event_mask = [[0, 0] for _ in range(int(max_events))]
    event_side = [0 for _ in range(int(max_events))]
    event_type = [0 for _ in range(int(max_events))]
    pick_index = 0
    for raw_event in DraftTimeline(row["events"]).events:
        event = coerce_action_event(raw_event)
        if event.action_type != "pick":
            raise ValueError("training state events must contain picks only")
        if pick_index >= int(max_events):
            raise ValueError("event timeline exceeds max_events")
        for slot, value in enumerate(event.monsters[:2]):
            monster_id = int(vocab.get_id(value))
            if monster_id <= 1 or monster_id >= len(vocab):
                raise ValueError(f"state contains a monster outside the vocabulary: {value}")
            event_ids[pick_index][slot] = monster_id
            event_mask[pick_index][slot] = 1
        event_side[pick_index] = 0 if event.side == "me" else 1
        event_type[pick_index] = 2 if len(event.monsters) == 2 else 1
        pick_index += 1
    return event_ids, event_mask, event_side, event_type


class SharedActionDataset(Dataset):
    """Materialize pick, mirrored-opponent, and ban training examples."""

    def __init__(self, pick_rows, ban_rows, vocab, max_events=10, negative_count=2, seed=42):
        self.vocab = vocab
        self.pad_id = int(vocab.get_id(PAD_TOKEN))
        self.valid_ids = [index for index in range(len(vocab)) if index > 1]
        self.max_events = int(max_events)
        self.negative_count = max(1, int(negative_count))
        self.seed = int(seed)
        self.rows = []
        for raw in pick_rows:
            self._append_pick(raw, head=0, mirror=False)
            self._append_pick(raw, head=1, mirror=True)
        for raw in ban_rows:
            self._append_ban(raw)
        self.encoded_states = [
            encode_state(
                {"events": row["events"]},
                self.vocab,
                self.max_events,
            )
            for row in self.rows
        ]
        self.negatives = []
        self.negative_masks = []
        self.refresh(self.seed)

    def _append_pick(self, raw, head, mirror):
        if head not in {0, 1} or mirror != (head == 1):
            raise ValueError("pick row head/mirror combination is invalid")
        required = ("events", "candidate", "is_pair", "turn", "first_pick_side", "outcome")
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"pick row missing canonical fields: {', '.join(missing)}")
        outcome = raw["outcome"]
        raw_events = raw["events"]
        timeline = DraftTimeline(raw_events)
        turn = int(raw["turn"])
        if turn < 1 or timeline.slots_used != turn - 1:
            raise ValueError("pick row turn does not match its event prefix")
        if any(event.action_type != "pick" for event in timeline.events):
            raise ValueError("pick row events must contain picks only")
        raw_my = list(timeline.picks_for("me"))
        raw_enemy = list(timeline.picks_for("enemy"))
        if mirror:
            own = raw_enemy
            enemy = raw_my
            first = 1.0 - _first_side_value(raw["first_pick_side"])
            if outcome is not None:
                outcome = 1.0 - float(outcome)
        else:
            own = raw_my
            enemy = raw_enemy
            first = _first_side_value(raw["first_pick_side"])
        if first not in {0.0, 1.0}:
            raise ValueError("first_pick_side must encode me/enemy as 1/0")
        if not isinstance(raw["is_pair"], bool):
            raise TypeError("pick row is_pair must be a boolean")
        is_pair = raw["is_pair"]
        candidate = tuple(int(value) for value in raw["candidate"])
        if any(value <= 1 or value >= len(self.vocab) for value in candidate if value != self.pad_id):
            raise ValueError("pick row candidate ID is outside the vocabulary")
        occupied_ids = {
            int(self.vocab.get_id(value))
            for value in own + enemy
        }
        if any(value <= 1 or value >= len(self.vocab) for value in occupied_ids):
            raise ValueError("pick row prefix contains a monster outside the vocabulary")
        if len(occupied_ids) != len(own) + len(enemy):
            raise ValueError("pick row prefix contains duplicate monsters")
        if any(value in occupied_ids for value in candidate if value != self.pad_id):
            raise ValueError("pick row candidate is already present in the prefix")
        if is_pair:
            if len(candidate) != 2:
                raise ValueError("pair row candidate must contain exactly two IDs")
            candidate = tuple(sorted(candidate))
            if candidate[0] <= 1 or candidate[0] == candidate[1] or candidate[1] <= 1:
                raise ValueError("pair row candidate IDs are invalid")
        else:
            if len(candidate) != 2 or candidate[0] <= 1 or candidate[1] != self.pad_id:
                raise ValueError("single row candidate must be (monster_id, pad_id)")
        expected_side = side_for_order(turn, "me" if first >= 0.5 else "enemy")
        expected_count = 0
        while (
            expected_side is not None
            and turn + expected_count <= 10
            and side_for_order(turn + expected_count, "me" if first >= 0.5 else "enemy") == expected_side
        ):
            expected_count += 1
        if expected_count not in {1, 2} or is_pair != (expected_count == 2):
            raise ValueError("pick row candidate size does not match the BP event")
        if head == 0 and expected_side != "me":
            raise ValueError("pick row head does not own the next event")
        if head == 1 and expected_side != "enemy":
            raise ValueError("mirrored pick row does not map to the opponent event")
        if outcome is not None and not 0.0 <= float(outcome) <= 1.0:
            raise ValueError("outcome must be in [0, 1]")
        events = self._row_events(
            raw_events,
            mirror=mirror,
        )
        self.rows.append(
            {
                "own": own,
                "enemy": enemy,
                "first": first,
                "turn": float(turn),
                "candidate": candidate,
                "is_pair": is_pair,
                "head": int(head),
                "outcome": None if outcome is None else float(outcome),
                "events": events,
            }
        )

    @staticmethod
    def _row_events(raw_events, *, mirror=False):
        timeline = DraftTimeline(raw_events)
        if not mirror:
            return timeline.to_dict()
        return [
            DraftActionEvent(
                side="enemy" if event.side == "me" else "me",
                monsters=event.monsters,
                action_type=event.action_type,
                order=event.order,
            ).to_dict()
            for event in timeline.events
        ]

    def _append_ban(self, raw):
        required = ("events", "candidate", "turn", "first_pick_side", "enemy_candidate_ids", "outcome")
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"ban row missing canonical fields: {', '.join(missing)}")
        raw_events = raw["events"]
        timeline = DraftTimeline(raw_events)
        if timeline.slots_used != 10 or any(event.action_type != "pick" for event in timeline.events):
            raise ValueError("ban row must contain the complete ten-slot pick timeline")
        if int(raw["turn"]) != 11:
            raise ValueError("ban row turn must be 11")
        own = list(timeline.picks_for("me"))
        enemy = list(timeline.picks_for("enemy"))
        if len(own) != 5 or len(enemy) != 5:
            raise ValueError("ban row must contain five picks per side")
        candidate = int(raw["candidate"])
        if candidate <= 1 or candidate >= len(self.vocab):
            raise ValueError("ban row candidate ID is invalid")
        ban_pool = [int(value) for value in raw["enemy_candidate_ids"]]
        enemy_ids = [int(self.vocab.get_id(value)) for value in enemy]
        if (
            len(ban_pool) != 5
            or len(set(ban_pool)) != 5
            or any(value <= 1 or value >= len(self.vocab) for value in ban_pool)
            or set(ban_pool) != set(enemy_ids)
            or candidate not in ban_pool
        ):
            raise ValueError("ban row candidate pool is invalid")
        first = _first_side_value(raw["first_pick_side"])
        outcome = raw["outcome"]
        if outcome is not None and not 0.0 <= float(outcome) <= 1.0:
            raise ValueError("outcome must be in [0, 1]")
        self.rows.append(
            {
                "own": own,
                "enemy": enemy,
                "first": first,
                "turn": 11.0,
                "candidate": (candidate,),
                "is_pair": False,
                "head": 2,
                "outcome": None if outcome is None else float(outcome),
                "ban_pool": ban_pool,
                "events": self._row_events(raw_events),
            }
        )

    def refresh(self, epoch_seed):
        rng = random.Random(int(self.seed) + int(epoch_seed))
        self.negatives = []
        self.negative_masks = []
        for row in self.rows:
            blocked = {
                int(self.vocab.get_id(value))
                for value in row["own"] + row["enemy"]
                if int(self.vocab.get_id(value)) > 1
            }
            legal = [value for value in self.valid_ids if value not in blocked]
            if row["head"] == 2 and row.get("ban_pool"):
                legal = [value for value in row["ban_pool"] if value != row["candidate"][0]]
            if row["is_pair"]:
                pool = [value for value in legal if value not in row["candidate"]]
                pairs = []
                for _ in range(self.negative_count * 8):
                    if len(pool) < 2:
                        break
                    first, second = rng.sample(pool, 2)
                    pair = tuple(sorted((first, second)))
                    if pair != row["candidate"] and pair not in pairs:
                        pairs.append(pair)
                    if len(pairs) >= self.negative_count:
                        break
                if len(pairs) != self.negative_count:
                    raise ValueError("not enough legal negative pairs for a training row")
                self.negatives.append(pairs)
                self.negative_masks.append([1] * len(pairs))
            else:
                choices = [value for value in legal if value != row["candidate"][0]]
                if len(choices) < self.negative_count:
                    raise ValueError("not enough legal negative singles for a training row")
                negatives = rng.sample(choices, self.negative_count)
                self.negatives.append(negatives)
                self.negative_masks.append([1] * self.negative_count)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        event_ids, event_mask, event_side, event_type = self.encoded_states[index]
        candidate = row["candidate"]
        negative_mask = list(self.negative_masks[index])
        if row["is_pair"]:
            negative_pairs = self.negatives[index]
            pairs = [candidate] + negative_pairs
            pair_ids = [list(pair) for pair in pairs]
            pair_ids += [[self.pad_id, self.pad_id]] * max(0, self.negative_count + 1 - len(pair_ids))
            negative_mask = [1] + negative_mask
        else:
            pair_ids = [[candidate[0], self.pad_id]]
            pair_ids += [[negative, self.pad_id] for negative in self.negatives[index]]
            pair_ids += [[self.pad_id, self.pad_id]] * max(
                0, self.negative_count + 1 - len(pair_ids)
            )
            negative_mask = [1] + negative_mask
        negative_mask += [0] * max(0, self.negative_count + 1 - len(negative_mask))
        return {
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "event_mask": torch.tensor(event_mask, dtype=torch.bool),
            "event_side": torch.tensor(event_side, dtype=torch.long),
            "event_type": torch.tensor(event_type, dtype=torch.long),
            "turn": torch.tensor(row["turn"], dtype=torch.float32),
            "first_pick_side": torch.tensor(row["first"], dtype=torch.float32),
            "current_side": torch.tensor(
                1.0 if row["head"] in {0, 2} else 0.0,
                dtype=torch.float32,
            ),
            "pair_ids": torch.tensor(pair_ids, dtype=torch.long),
            "negative_mask": torch.tensor(negative_mask, dtype=torch.bool),
            "candidate": torch.tensor(candidate[0], dtype=torch.long),
            "is_pair": torch.tensor(int(row["is_pair"]), dtype=torch.bool),
            "head": torch.tensor(row["head"], dtype=torch.long),
            "outcome": torch.tensor(0.0 if row["outcome"] is None else row["outcome"], dtype=torch.float32),
            "outcome_mask": torch.tensor(int(row["outcome"] is not None and row["head"] == 0), dtype=torch.float32),
        }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _pairwise_loss(scores, negative_mask=None):
    if scores.numel() == 0:
        return scores.new_zeros(())
    losses = F.softplus(-(scores[:, :1] - scores[:, 1:]))
    if negative_mask is not None:
        mask = negative_mask[:, 1:].bool()
        if not bool(mask.any()):
            return scores.new_zeros(())
        return losses[mask].mean()
    return losses.mean()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    totals = {
        "loss": 0.0,
        "count": 0,
        "single_correct": 0,
        "single_total": 0,
        "opponent_correct": 0,
        "opponent_total": 0,
        "pair_correct": 0,
        "pair_total": 0,
        "ban_correct": 0,
        "ban_total": 0,
    }
    iterator = tqdm(loader, desc="shared train" if train else "shared valid", leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["event_ids"],
            batch["event_mask"],
            batch["event_side"],
            batch["event_type"],
            turn=batch["turn"],
            first_pick_side=batch["first_pick_side"],
            current_side=batch["current_side"],
            pair_candidates=batch["pair_ids"],
        )
        pair_mask = batch["is_pair"]
        single_mask = ~pair_mask
        pair_loss = _pairwise_loss(
            output["pair_logits"][pair_mask],
            batch["negative_mask"][pair_mask],
        )
        losses = []
        for head, logits in ((0, output["pick_logits"]), (1, output["opponent_logits"]), (2, output["ban_logits"])):
            mask = single_mask & (batch["head"] == head)
            if not bool(mask.any()):
                continue
            positives = logits[mask].gather(1, batch["candidate"][mask].unsqueeze(1)).squeeze(1)
            negative_ids = batch["pair_ids"][mask, 1:, 0]
            negative = logits[mask].gather(1, negative_ids)
            losses.append(
                _pairwise_loss(
                    torch.cat([positives.unsqueeze(1), negative], dim=1),
                    batch["negative_mask"][mask],
                )
            )
        value_mask = batch["outcome_mask"] > 0
        if bool(value_mask.any()):
            value_loss = F.binary_cross_entropy_with_logits(
                output["value"][value_mask], batch["outcome"][value_mask]
            )
        else:
            value_loss = output["value"].new_zeros(())
        loss = (sum(losses) / max(1, len(losses))) + pair_loss + 0.25 * value_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            if bool(single_mask.any()):
                for head, logits, target in (
                    (0, output["pick_logits"], "single_correct"),
                    (1, output["opponent_logits"], "opponent_correct"),
                    (2, output["ban_logits"], "ban_correct"),
                ):
                    mask = single_mask & (batch["head"] == head)
                    if bool(mask.any()):
                        positive_scores = logits[mask].gather(
                            1, batch["candidate"][mask].unsqueeze(1)
                        )
                        negative_ids = batch["pair_ids"][mask, 1:, 0]
                        negative_scores = logits[mask].gather(1, negative_ids)
                        negative_valid = batch["negative_mask"][mask, 1:].bool()
                        row_valid = negative_valid.any(dim=1)
                        if bool(row_valid.any()):
                            candidate_scores = torch.cat(
                                [positive_scores, negative_scores], dim=1
                            )
                            candidate_scores[:, 1:] = candidate_scores[:, 1:].masked_fill(
                                ~negative_valid, -float("inf")
                            )
                            totals[target] += int(
                                (candidate_scores[row_valid].argmax(dim=1) == 0).sum()
                            )
                            totals[target.replace("correct", "total")] += int(row_valid.sum())
            if bool(pair_mask.any()):
                pair_scores = output["pair_logits"][pair_mask].clone()
                pair_valid = batch["negative_mask"][pair_mask][:, 1:].bool()
                row_valid = pair_valid.any(dim=1)
                if bool(row_valid.any()):
                    pair_scores[:, 1:] = pair_scores[:, 1:].masked_fill(
                        ~pair_valid, -float("inf")
                    )
                    totals["pair_correct"] += int(
                        (pair_scores[row_valid].argmax(dim=1) == 0).sum()
                    )
                    totals["pair_total"] += int(row_valid.sum())
        count = int(batch["head"].numel())
        totals["loss"] += float(loss.detach()) * count
        totals["count"] += count
        iterator.set_postfix(loss=f"{float(loss.detach()):.3f}")
    count = max(1, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "single_top1": totals["single_correct"] / max(1, totals["single_total"]),
        "opponent_top1": totals["opponent_correct"] / max(1, totals["opponent_total"]),
        "pair_top1": totals["pair_correct"] / max(1, totals["pair_total"]),
        "ban_top1": totals["ban_correct"] / max(1, totals["ban_total"]),
        "single_total": totals["single_total"],
        "opponent_total": totals["opponent_total"],
        "pair_total": totals["pair_total"],
        "ban_total": totals["ban_total"],
        "count": totals["count"],
    }


def prepare_samples(path):
    samples = load_samples(path)
    if not samples:
        raise RuntimeError(f"没有读取到训练数据：{path}")
    print(f"[shared] loaded samples: {len(samples)}", flush=True)
    resources = build_name_resolution_resources(samples)
    resolver = NameResolver(
        alias_map=resources["name_slug_aliases"],
        display_name_map=resources["display_name_map"],
    )
    samples = [
        normalize_bp_sample(sample, resolver)
        for sample in tqdm(samples, desc="shared resolve aliases", unit="sample")
    ]
    return samples, resources


def save_artifacts(model, vocab, args, metrics, sample_counts, resources=None):
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = prefix.with_suffix(".pt")
    meta_path = prefix.with_name(prefix.name + "_meta.json")
    vocab_path = prefix.with_name(prefix.name + "_vocab.json")
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "w", encoding="utf-8") as file:
        json.dump(vocab.to_dict(), file, ensure_ascii=False, indent=2)
    meta = {
        "architecture": "shared_event_v1",
        "architecture_version": 1,
        "model_path": str(model_path),
        "embed_dim": model.monster_encoder.embed_dim,
        "state_dim": model.state_encoder.state_dim,
        "max_events": model.state_encoder.max_events,
        "slots_per_event": model.state_encoder.slots_per_event,
        "hidden_dim": model.state_encoder.state_projection[0].out_features,
        "num_heads": 4,
        "num_layers": len(model.state_encoder.event_transformer.layers),
        "ffn_dim": model.state_encoder.event_transformer.layers[0].linear1.out_features,
        "dropout": 0.15,
        "display_name_map": (resources or {}).get("display_name_map", {}),
        "name_slug_aliases": (resources or {}).get("name_slug_aliases", {}),
        "sample_counts": sample_counts,
        "metrics": metrics,
        "seed": int(args.seed),
        "split_manifest_path": str(prefix.with_name(prefix.name + "_split_manifest.json")),
    }
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return model_path, meta_path, vocab_path


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    samples, resources = prepare_samples(args.data_path)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]
    train_samples, valid_samples = split_samples(samples, args.val_ratio, args.seed)
    save_split_manifest(
        samples,
        train_samples,
        valid_samples,
        Path(args.output_prefix).with_name(Path(args.output_prefix).name + "_split_manifest.json"),
        args.seed,
        args.val_ratio,
    )
    vocab = build_vocab(samples)
    train_pick = build_positive_actions(train_samples, vocab)
    valid_pick = build_positive_actions(valid_samples, vocab)
    train_ban = build_ban_positive_actions(train_samples, vocab)
    valid_ban = build_ban_positive_actions(valid_samples, vocab)
    train_set = SharedActionDataset(
        train_pick,
        train_ban,
        vocab,
        max_events=args.max_events,
        negative_count=args.negative_count,
        seed=args.seed,
    )
    valid_set = SharedActionDataset(
        valid_pick,
        valid_ban,
        vocab,
        max_events=args.max_events,
        negative_count=args.negative_count,
        seed=args.seed + 1,
    )
    if not len(train_set):
        raise RuntimeError("没有生成可训练的 BP action 样本，请先审计/同步数据")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BPSharedNetwork(
        vocab_size=len(vocab),
        max_events=args.max_events,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )
    best = None
    best_state = None
    for epoch in range(1, max(1, int(args.epochs)) + 1):
        train_set.refresh(args.seed + epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, None, device, train=False) if len(valid_set) else {}
        # Weight each head by the number of validation examples it actually
        # evaluated.  This keeps model selection stable when a snapshot has
        # few pair or ban records.
        metric_parts = []
        for metric_name, count_name in (
            ("single_top1", "single_total"),
            ("opponent_top1", "opponent_total"),
            ("pair_top1", "pair_total"),
            ("ban_top1", "ban_total"),
        ):
            count = int(valid_metrics.get(count_name, 0) or 0)
            if count > 0:
                metric_parts.append((float(valid_metrics.get(metric_name, 0.0)), count))
        score = (
            sum(value * count for value, count in metric_parts)
            / max(1, sum(count for _value, count in metric_parts))
            if metric_parts
            else -float(train_metrics.get("loss", 0.0))
        )
        print(
            f"[shared] epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_single={valid_metrics.get('single_top1', 0.0):.4f} "
            f"val_pair={valid_metrics.get('pair_top1', 0.0):.4f} "
            f"val_opponent={valid_metrics.get('opponent_top1', 0.0):.4f} "
            f"val_ban={valid_metrics.get('ban_top1', 0.0):.4f}",
            flush=True,
        )
        if best is None or score >= best:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "train": train_metrics,
        "valid": valid_metrics,
    }
    paths = save_artifacts(
        model,
        vocab,
        args,
        metrics,
        {
            "all": len(samples),
            "train": len(train_samples),
            "valid": len(valid_samples),
            "train_actions": len(train_set),
            "valid_actions": len(valid_set),
        },
        resources=resources,
    )
    print("[shared] saved:", *(str(path) for path in paths), flush=True)


if __name__ == "__main__":
    main()
