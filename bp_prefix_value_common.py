import torch
import torch.nn as nn

from bp_relation_common import MonsterVocab
from bp_transformer_common import PAD_TOKEN, SequenceFeatureTransformer


class BPPrefixValueNet(nn.Module):
    """Value model trained on chronological partial-draft states."""

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
        self.pad_idx = int(pad_idx)
        self.backbone = SequenceFeatureTransformer(
            vocab_size=vocab_size,
            role_vocab_size=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            pad_idx=pad_idx,
            max_seq_len=16,
            numeric_dim=3,
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
        turn,
        first_pick_side,
        current_side,
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
        numeric_features = torch.stack(
            [
                turn.float() / 10.0,
                first_pick_side.float(),
                current_side.float(),
            ],
            dim=-1,
        )
        _, cls_vec = self.backbone(
            input_ids=input_ids,
            valid_mask=valid_mask,
            role_ids=role_ids,
            numeric_features=numeric_features,
        )
        hidden = self.head(cls_vec)
        return self.value_head(hidden).squeeze(-1)


def build_prefix_value_vocab(samples):
    vocab = MonsterVocab()
    for sample in samples or []:
        for key in (
            "my_picks_slugs",
            "enemy_picks_slugs",
            "my_leader_slug",
            "enemy_leader_slug",
        ):
            values = sample.get(key)
            if isinstance(values, list):
                for value in values:
                    vocab.add_token(value)
            elif values:
                vocab.add_token(values)
        for key in ("my_picks", "enemy_picks", "my_leader", "enemy_leader"):
            values = sample.get(key)
            if isinstance(values, list):
                for value in values:
                    vocab.add_token(value)
            elif values:
                vocab.add_token(values)
    return vocab


def encode_prefix_list(values, vocab, max_len=5):
    ids = [vocab.get_id(value) for value in (values or [])[:max_len]]
    mask = [1.0] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab.get_id(PAD_TOKEN))
        mask.append(0.0)
    return ids, mask
