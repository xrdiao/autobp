from __future__ import annotations

from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - allows non-training utilities to import this module.
    torch = None

    class _NNModule:
        pass

    class _NNNamespace:
        Module = _NNModule

    nn = _NNNamespace()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def normalize_name(name):
    if name is None:
        return ""
    return str(name).strip().lower()


def slugify_fallback(name):
    text = normalize_name(name)
    text = text.replace(" - ", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    return "-".join(text.split())


class SequenceFeatureTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        role_vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        ffn_dim=384,
        dropout=0.15,
        pad_idx=0,
        max_seq_len=16,
        numeric_dim=0,
    ):
        if torch is None:
            raise ModuleNotFoundError("torch is required to instantiate SequenceFeatureTransformer")
        super().__init__()
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.numeric_dim = numeric_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.role_embedding = nn.Embedding(role_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if numeric_dim > 0:
            self.numeric_proj = nn.Sequential(
                nn.Linear(numeric_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.numeric_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        if torch is None:
            raise ModuleNotFoundError("torch is required to use SequenceFeatureTransformer")
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, input_ids, valid_mask, role_ids, numeric_features=None):
        if torch is None:
            raise ModuleNotFoundError("torch is required to use SequenceFeatureTransformer")
        batch_size, token_len = input_ids.shape
        if token_len + 1 > self.max_seq_len:
            raise ValueError(f"sequence too long: {token_len + 1} > max_seq_len={self.max_seq_len}")

        device = input_ids.device
        valid_mask = valid_mask.bool()
        role_ids = role_ids.to(device=device, dtype=torch.long)

        token_emb = self.token_embedding(input_ids)
        role_emb = self.role_embedding(role_ids)
        seq = token_emb + role_emb

        cls = self.cls_token.expand(batch_size, -1, -1)
        if self.numeric_proj is not None and numeric_features is not None:
            cls = cls + self.numeric_proj(numeric_features).unsqueeze(1)

        seq = torch.cat([cls, seq], dim=1)
        pos_ids = torch.arange(seq.size(1), device=device, dtype=torch.long).unsqueeze(0)
        seq = seq + self.position_embedding(pos_ids)
        seq = self.dropout(self.input_norm(seq))

        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                ~valid_mask,
            ],
            dim=1,
        )

        encoded = self.encoder(seq, src_key_padding_mask=padding_mask)
        encoded = self.output_norm(encoded)
        return encoded, encoded[:, 0]
