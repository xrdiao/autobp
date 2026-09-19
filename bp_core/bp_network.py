"""Shared neural components for objective BP recommendation.

The module is the neural core used by the Shared BP recommender.  It
provides the shared backbone used by the event-aware training and inference path:

* monster tokens can combine an ID embedding with optional structured data;
* the draft is an ordered sequence of *events* (a pair event contains an
  unordered set of two monsters);
* team composition is encoded with permutation-invariant pooling;
* one state representation feeds pick, pair, opponent, ban and value heads.

The canonical event tensors have these shapes::

    event_ids:       [batch, events, slots_per_event]
    event_mask:      [batch, events, slots_per_event] (bool or 0/1)
    event_side:      [batch, events] (0 = me, 1 = enemy)
    event_type:      [batch, events] (0 = padding, 1 = single, 2 = pair)

PyTorch is a required project dependency and is pinned in environment.yml.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from bp_core.bp_common import MonsterVocab
from bp_core.names import PAD_TOKEN, UNK_TOKEN, normalize_name


EVENT_PAD = 0
EVENT_SINGLE = 1
EVENT_PAIR = 2



def _check_last_dim(name: str, value, expected: int):
    if value is not None and value.size(-1) != int(expected):
        raise ValueError(f"{name} has last dimension {value.size(-1)}, expected {expected}")


def _as_tensor(value, *, device=None, dtype=None):
    """Convert public list/array inputs without copying existing tensors."""

    if torch.is_tensor(value):
        if device is not None or dtype is not None:
            return value.to(device=device, dtype=dtype)
        return value
    return torch.as_tensor(value, device=device, dtype=dtype)


class MonsterFeatureEncoder(nn.Module):
    """Encode monster IDs and optional catalog/statistical features.

    Optional tensors are additive and can be omitted at inference time.  The
    expected fields are ``element_ids``, ``family_ids``, ``mechanic_features``
    and ``stat_features``.  Categorical IDs use zero for padding; unknown
    values must be resolved by the data pipeline before encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        *,
        element_vocab_size: int = 0,
        family_vocab_size: int = 0,
        mechanic_dim: int = 0,
        stat_dim: int = 0,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.pad_idx = int(pad_idx)
        self.mechanic_dim = int(mechanic_dim)
        self.stat_dim = int(stat_dim)

        self.id_embedding = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.element_embedding = (
            nn.Embedding(int(element_vocab_size), self.embed_dim, padding_idx=0)
            if int(element_vocab_size) > 0
            else None
        )
        self.family_embedding = (
            nn.Embedding(int(family_vocab_size), self.embed_dim, padding_idx=0)
            if int(family_vocab_size) > 0
            else None
        )
        # Bias-free projections make an explicitly zero feature truly neutral.
        self.mechanic_projection = (
            nn.Linear(self.mechanic_dim, self.embed_dim, bias=False)
            if self.mechanic_dim > 0
            else None
        )
        self.stat_projection = (
            nn.Linear(self.stat_dim, self.embed_dim, bias=False)
            if self.stat_dim > 0
            else None
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        monster_ids,
        *,
        element_ids=None,
        family_ids=None,
        mechanic_features=None,
        stat_features=None,
        apply_dropout=True,
    ):
        model_device = next(self.parameters()).device
        monster_ids = _as_tensor(monster_ids, device=model_device, dtype=torch.long)
        if monster_ids.dtype != torch.long:
            monster_ids = monster_ids.long()
        encoded = self.id_embedding(monster_ids)

        if element_ids is not None:
            if self.element_embedding is None:
                raise ValueError("element_ids were supplied but element_vocab_size is zero")
            element_ids = _as_tensor(element_ids, device=model_device, dtype=torch.long)
            encoded = encoded + self.element_embedding(element_ids)
        if family_ids is not None:
            if self.family_embedding is None:
                raise ValueError("family_ids were supplied but family_vocab_size is zero")
            family_ids = _as_tensor(family_ids, device=model_device, dtype=torch.long)
            encoded = encoded + self.family_embedding(family_ids)
        if mechanic_features is not None:
            if self.mechanic_projection is None:
                raise ValueError("mechanic_features were supplied but mechanic_dim is zero")
            mechanic_features = _as_tensor(mechanic_features, device=model_device)
            _check_last_dim("mechanic_features", mechanic_features, self.mechanic_dim)
            encoded = encoded + self.mechanic_projection(mechanic_features.float())
        if stat_features is not None:
            if self.stat_projection is None:
                raise ValueError("stat_features were supplied but stat_dim is zero")
            stat_features = _as_tensor(stat_features, device=model_device)
            _check_last_dim("stat_features", stat_features, self.stat_dim)
            encoded = encoded + self.stat_projection(stat_features.float())

        encoded = self.output_norm(encoded)
        return self.dropout(encoded) if apply_dropout else encoded


class _MaskedSetPool(nn.Module):
    """Permutation-invariant mean/max pooling with an empty-set result of 0."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(int(input_dim) * 2, int(output_dim)),
            nn.GELU(),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, values, mask):
        mask = mask.bool()
        if values.size(-2) == 0:
            return values.new_zeros((*values.shape[:-2], self.output_dim))
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        count = weights.sum(dim=-2).clamp_min(1.0)
        mean = (values * weights).sum(dim=-2) / count
        neg_inf = torch.finfo(values.dtype).min
        maximum = values.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=-2)
        has_value = mask.any(dim=-1, keepdim=True)
        maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
        pooled = self.projection(torch.cat([mean, maximum], dim=-1))
        return pooled * has_value.to(dtype=pooled.dtype)


class BPStateEncoder(nn.Module):
    """Encode ordered BP events and current team sets into one state vector."""

    def __init__(
        self,
        monster_encoder: MonsterFeatureEncoder,
        *,
        state_dim: int = 256,
        hidden_dim: int = 256,
        max_events: int = 10,
        slots_per_event: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        if int(slots_per_event) < 2:
            raise ValueError("slots_per_event must be at least 2")
        self.monster_encoder = monster_encoder
        self.embed_dim = int(monster_encoder.embed_dim)
        self.state_dim = int(state_dim)
        self.max_events = int(max_events)
        self.slots_per_event = int(slots_per_event)

        self.event_pool = _MaskedSetPool(self.embed_dim, self.embed_dim)
        self.team_pool = _MaskedSetPool(self.embed_dim, self.embed_dim)
        self.position_embedding = nn.Embedding(self.max_events, self.embed_dim)
        self.side_embedding = nn.Embedding(2, self.embed_dim)
        self.type_embedding = nn.Embedding(3, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.input_norm = nn.LayerNorm(self.embed_dim)
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Explicitly disable nested tensors: norm_first=True intentionally
        # disables that optimization in PyTorch and otherwise emits a warning.
        self.event_transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(3, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(self.embed_dim * 3 + int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.state_dim),
            nn.GELU(),
            nn.LayerNorm(self.state_dim),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _encode_monster_features(self, ids, features: Mapping[str, object] | None = None):
        features = dict(features or {})
        features["apply_dropout"] = False
        return self.monster_encoder(ids, **features)

    def forward(
        self,
        event_ids,
        event_mask,
        event_side,
        event_type,
        *,
        turn,
        first_pick_side,
        current_side,
        event_features: Mapping[str, object] | None = None,
    ):
        model_device = next(self.parameters()).device
        event_ids = _as_tensor(event_ids, device=model_device, dtype=torch.long)
        event_mask = _as_tensor(event_mask, device=model_device)
        event_side = _as_tensor(event_side, device=model_device)
        if event_ids.ndim != 3:
            raise ValueError("event_ids must have shape [batch, events, slots]")
        batch_size, event_count, slot_count = event_ids.shape
        if event_count > self.max_events:
            raise ValueError(f"too many events: {event_count} > max_events={self.max_events}")
        if slot_count != self.slots_per_event:
            raise ValueError(f"expected {self.slots_per_event} slots per event, got {slot_count}")
        if bool((event_ids < 0).any()) or bool((event_ids >= self.monster_encoder.vocab_size).any()):
            raise ValueError("event_ids contain an ID outside the vocabulary")
        event_mask = event_mask.bool()
        if event_mask.shape != event_ids.shape:
            raise ValueError("event_mask must have shape [batch, events, slots]")
        event_side = event_side.long()
        if event_side.shape != (batch_size, event_count):
            raise ValueError("event_side must have shape [batch, events]")
        if bool(((event_side < 0) | (event_side > 1)).any()):
            raise ValueError("event_side values must be 0 or 1")
        event_type = _as_tensor(event_type, device=model_device, dtype=torch.long)
        if event_type.shape != (batch_size, event_count):
            raise ValueError("event_type must have shape [batch, events]")
        if bool(((event_type < EVENT_PAD) | (event_type > EVENT_PAIR)).any()):
            raise ValueError("event_type values must be 0, 1, or 2")
        slot_counts = event_mask.sum(dim=-1)
        if bool((slot_counts > 2).any()):
            raise ValueError("each BP event can contain at most two occupied slots")
        if event_mask.size(-1) > 1 and bool(
            (event_mask[..., 1:] & ~event_mask[..., :-1]).any()
        ):
            raise ValueError("event_mask slots must be occupied from left to right")
        expected_type = torch.where(
            slot_counts == 0,
            torch.full_like(slot_counts, EVENT_PAD),
            torch.where(
                slot_counts == 1,
                torch.full_like(slot_counts, EVENT_SINGLE),
                torch.full_like(slot_counts, EVENT_PAIR),
            ),
        )
        if not torch.equal(event_type, expected_type):
            raise ValueError("event_type must match event_mask slot counts")
        if bool(((event_type == EVENT_PAD) & (event_side != 0)).any()):
            raise ValueError("padding events must use side 0")
        if bool((event_mask & (event_ids <= 1)).any()):
            raise ValueError("occupied event slots must contain a known monster ID")
        if bool(((~event_mask) & (event_ids != self.monster_encoder.pad_idx)).any()):
            raise ValueError("empty event slots must contain the vocabulary pad ID")
        event_valid = event_mask.any(dim=-1)

        flat_features = {}
        for name, value in (event_features or {}).items():
            if value is None:
                continue
            value = _as_tensor(value, device=event_ids.device)
            if value.ndim >= 3 and value.shape[0] == 1 and batch_size != 1:
                value = value.expand(batch_size, *value.shape[1:])
            if value.shape[:3] != (batch_size, event_count, slot_count):
                raise ValueError(f"event feature {name} must start with [batch, events, slots]")
            flat_features[name] = value.reshape(batch_size * event_count, slot_count, *value.shape[3:])
        monster_features = self._encode_monster_features(
            event_ids.reshape(batch_size * event_count, slot_count), flat_features
        ).reshape(batch_size, event_count, slot_count, self.embed_dim)

        event_tokens = self.event_pool(
            monster_features.reshape(batch_size * event_count, slot_count, self.embed_dim),
            event_mask.reshape(batch_size * event_count, slot_count),
        ).reshape(batch_size, event_count, self.embed_dim)

        # Pool all picks by side separately.  The operation is invariant to the
        # order inside a pair, while the Transformer below still sees event order.
        my_mask = event_mask & (event_side.unsqueeze(-1) == 0)
        enemy_mask = event_mask & (event_side.unsqueeze(-1) == 1)
        flat_monsters = monster_features.reshape(batch_size, event_count * slot_count, self.embed_dim)
        my_team = self.team_pool(flat_monsters, my_mask.reshape(batch_size, event_count * slot_count))
        enemy_team = self.team_pool(flat_monsters, enemy_mask.reshape(batch_size, event_count * slot_count))

        device = event_ids.device
        positions = torch.arange(event_count, device=device, dtype=torch.long).unsqueeze(0)
        event_tokens = (
            event_tokens
            + self.position_embedding(positions)
            + self.side_embedding(event_side)
            + self.type_embedding(event_type)
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = self.dropout(self.input_norm(torch.cat([cls, event_tokens], dim=1)))
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                ~event_valid,
            ],
            dim=1,
        )
        encoded = self.output_norm(
            self.event_transformer(sequence, src_key_padding_mask=padding_mask)
        )

        def _numeric(value, name):
            if isinstance(value, str):
                text = value.strip().lower()
                if text not in {"me", "my", "own", "enemy", "opponent", "foe", "them"}:
                    raise ValueError(f"{name} must be numeric or a valid side name")
                value = 1.0 if text in {"me", "my", "own"} else 0.0
            value = torch.as_tensor(value, device=device).float()
            if value.numel() == 1:
                return value.expand(batch_size)
            if value.numel() != batch_size:
                raise ValueError(f"{name} must contain one value per batch item")
            return value.reshape(batch_size)

        turn_value = _numeric(turn, "turn") / 10.0
        if bool((turn_value < 0.1).any()) or bool((turn_value > 1.1).any()):
            raise ValueError("turn must be in the range 1..11")
        first_value = _numeric(first_pick_side, "first_pick_side")
        current_value = _numeric(current_side, "current_side")
        if bool(((first_value < 0.0) | (first_value > 1.0)).any()):
            raise ValueError("first_pick_side must be encoded as 0 or 1")
        if bool(((current_value < 0.0) | (current_value > 1.0)).any()):
            raise ValueError("current_side must be encoded as 0 or 1")
        numeric = torch.stack(
            [
                turn_value,
                first_value,
                current_value,
            ],
            dim=-1,
        )
        numeric = self.numeric_projection(numeric)
        state = self.state_projection(torch.cat([encoded[:, 0], my_team, enemy_team, numeric], dim=-1))
        return {
            "state": state,
            "event_tokens": encoded[:, 1:],
            "team_tokens": torch.stack([my_team, enemy_team], dim=1),
            "event_padding_mask": padding_mask[:, 1:],
        }


class BPSharedNetwork(nn.Module):
    """Shared BP encoder with pick, pair, opponent, ban and value heads.

    ``forward`` returns a dictionary containing ``pick_logits`` (single-pick
    logits over the vocabulary), ``opponent_logits``, ``ban_logits``,
    ``value`` and optional ``pair_logits``.  Pair scoring is symmetric by
    construction: swapping the two IDs produces exactly the same score in
    evaluation mode (and the same function in training mode before dropout).
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 128,
        state_dim: int = 256,
        hidden_dim: int = 256,
        element_vocab_size: int = 0,
        family_vocab_size: int = 0,
        mechanic_dim: int = 0,
        stat_dim: int = 0,
        dropout: float = 0.15,
        pad_idx: int = 0,
        max_events: int = 10,
        slots_per_event: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = int(pad_idx)
        self.embed_dim = int(embed_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_events = int(max_events)
        self.slots_per_event = int(slots_per_event)
        monster_encoder = MonsterFeatureEncoder(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            element_vocab_size=element_vocab_size,
            family_vocab_size=family_vocab_size,
            mechanic_dim=mechanic_dim,
            stat_dim=stat_dim,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.state_encoder = BPStateEncoder(
            monster_encoder,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            max_events=max_events,
            slots_per_event=slots_per_event,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pick_head = nn.Linear(state_dim, self.vocab_size)
        self.opponent_head = nn.Linear(state_dim, self.vocab_size)
        self.ban_head = nn.Linear(state_dim, self.vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.action_query = nn.Linear(state_dim, embed_dim)
        self.pair_action_key = nn.Linear(embed_dim * 3, embed_dim)
        self.pair_scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def monster_encoder(self):
        """Expose the shared encoder without registering it twice in state_dict."""
        return self.state_encoder.monster_encoder

    @staticmethod
    def _expand_candidates(candidate_ids, batch_size: int):
        if candidate_ids.ndim == 1:
            return candidate_ids.unsqueeze(0).expand(batch_size, -1)
        if candidate_ids.ndim != 2:
            raise ValueError("candidate IDs must have shape [N] or [batch, N]")
        if candidate_ids.size(0) == 1 and batch_size != 1:
            return candidate_ids.expand(batch_size, -1)
        if candidate_ids.size(0) != batch_size:
            raise ValueError("candidate batch dimension does not match state batch")
        return candidate_ids

    def score_single_candidates(self, state, candidate_ids):
        """Return single-pick logits for a candidate subset."""
        ids = self._expand_candidates(
            _as_tensor(candidate_ids, device=state.device, dtype=torch.long),
            state.size(0),
        )
        return self.pick_head(state).gather(1, ids)

    def _pair_embeddings(self, pair_ids, pair_features=None):
        if pair_ids.ndim != 3 or pair_ids.size(-1) != 2:
            raise ValueError("pair IDs must have shape [batch, pairs, 2]")
        batch_size, pair_count, _ = pair_ids.shape
        features = {}
        for name, value in (pair_features or {}).items():
            if value is None:
                continue
            if name == "apply_dropout":
                continue
            value = _as_tensor(value, device=pair_ids.device)
            if value.shape[:3] != (batch_size, pair_count, 2):
                raise ValueError(f"pair feature {name} must start with [batch, pairs, 2]")
            features[name] = value
        encoded = self.monster_encoder(
            pair_ids.reshape(batch_size * pair_count, 2),
            apply_dropout=False,
            **{name: value.reshape(batch_size * pair_count, 2, *value.shape[3:]) for name, value in features.items()},
        ).reshape(batch_size, pair_count, 2, -1)
        first, second = encoded[..., 0, :], encoded[..., 1, :]
        # Every operation is symmetric in first/second.
        return torch.cat([first + second, first * second, torch.abs(first - second)], dim=-1)

    def score_pair_candidates(self, state, pair_ids, pair_features=None):
        """Score unordered candidate pairs; ``(a,b)`` equals ``(b,a)``."""
        pair_ids = _as_tensor(pair_ids, device=state.device, dtype=torch.long)
        if pair_ids.numel() == 0 and pair_ids.ndim < 2:
            return state.new_empty((state.size(0), 0))
        pair_features = {
            name: (_as_tensor(value, device=state.device) if value is not None else None)
            for name, value in (pair_features or {}).items()
        }
        if pair_ids.ndim == 2:
            pair_count = pair_ids.size(0)
            pair_ids = pair_ids.unsqueeze(0).expand(state.size(0), -1, -1)
            for name, value in list(pair_features.items()):
                if value is not None and value.shape[:2] == (pair_count, 2):
                    pair_features[name] = value.unsqueeze(0).expand(state.size(0), -1, -1, *value.shape[2:])
        elif pair_ids.ndim == 3 and pair_ids.size(0) == 1 and state.size(0) != 1:
            pair_ids = pair_ids.expand(state.size(0), -1, -1)
            for name, value in list(pair_features.items()):
                if value is not None and value.size(0) == 1:
                    pair_features[name] = value.expand(state.size(0), -1, -1, *value.shape[3:])
        if pair_ids.size(0) != state.size(0):
            raise ValueError("pair batch dimension does not match state batch")
        # Keep the public scoring API well-defined for an empty candidate
        # pool.  The encoder reshape below cannot infer its final dimension
        # from a zero-element tensor, while callers still expect one empty
        # score row per state.
        if pair_ids.size(1) == 0:
            return state.new_empty((state.size(0), 0))
        pair_embedding = self._pair_embeddings(pair_ids, pair_features=pair_features)
        state_expanded = state.unsqueeze(1).expand(-1, pair_embedding.size(1), -1)
        query = self.action_query(state_expanded)
        key = self.pair_action_key(pair_embedding)
        interaction = query * key
        return self.pair_scorer(
            torch.cat([state_expanded, pair_embedding, interaction], dim=-1)
        ).squeeze(-1)

    @staticmethod
    def timeline_to_tensors(timeline, vocab, *, max_events=10, device=None):
        """Convert a ``DraftTimeline`` into one batched event tensor.

        The conversion keeps event order and puts the two members of a pair
        in the two event slots.  Ban events are omitted from the pick-state
        sequence; the separate ban head consumes the resulting state.
        ``vocab`` must be the canonical ``MonsterVocab`` used by training.
        """
        from bp_core.action_space import DraftTimeline, coerce_action_event

        if not isinstance(timeline, DraftTimeline):
            raise TypeError("timeline_to_tensors requires a DraftTimeline")
        events = timeline.events
        pick_events = []
        for raw_event in events:
            event = coerce_action_event(raw_event)
            if event.action_type == "pick":
                pick_events.append(event)
        if len(pick_events) > int(max_events):
            raise ValueError(f"too many pick events: {len(pick_events)} > max_events={max_events}")

        if not isinstance(vocab, MonsterVocab):
            raise TypeError("timeline_to_tensors requires the canonical MonsterVocab")

        def get_id(value):
            normalized = normalize_name(value)
            if not normalized:
                raise KeyError("timeline contains an empty monster")
            monster_id = int(vocab.get_id(value))
            if monster_id <= 1 and normalized not in {
                normalize_name(PAD_TOKEN),
                normalize_name(UNK_TOKEN),
            }:
                raise KeyError(f"monster is outside the vocabulary: {value}")
            return monster_id

        ids = torch.zeros((1, int(max_events), 2), dtype=torch.long, device=device)
        mask = torch.zeros((1, int(max_events), 2), dtype=torch.bool, device=device)
        sides = torch.zeros((1, int(max_events)), dtype=torch.long, device=device)
        types = torch.zeros((1, int(max_events)), dtype=torch.long, device=device)
        for index, event in enumerate(pick_events):
            values = tuple(event.monsters)
            for slot, value in enumerate(values[:2]):
                ids[0, index, slot] = get_id(value)
                mask[0, index, slot] = True
            sides[0, index] = 0 if event.side == "me" else 1
            types[0, index] = EVENT_PAIR if len(values) == 2 else EVENT_SINGLE
        return ids, mask, sides, types

    def forward_timeline(
        self,
        timeline,
        vocab,
        *,
        turn,
        first_pick_side,
        current_side,
        max_events=None,
        pair_candidates=None,
        pair_features=None,
    ):
        """Run the shared network directly from a ``DraftTimeline``."""
        max_events = self.state_encoder.max_events if max_events is None else int(max_events)
        events = self.timeline_to_tensors(
            timeline,
            vocab,
            max_events=max_events,
            device=next(self.parameters()).device,
        )
        return self.forward(
            *events,
            turn=turn,
            first_pick_side=first_pick_side,
            current_side=current_side,
            pair_candidates=pair_candidates,
            pair_features=pair_features,
        )

    def forward(
        self,
        event_ids,
        event_mask,
        event_side,
        event_type,
        *,
        turn,
        first_pick_side,
        current_side,
        event_features: Mapping[str, object] | None = None,
        pair_candidates=None,
        pair_features=None,
    ):
        encoded = self.state_encoder(
            event_ids,
            event_mask,
            event_side,
            event_type,
            turn=turn,
            first_pick_side=first_pick_side,
            current_side=current_side,
            event_features=event_features,
        )
        state = encoded["state"]
        output = {
            **encoded,
            "state": state,
            "pick_logits": self.pick_head(state),
            "opponent_logits": self.opponent_head(state),
            "ban_logits": self.ban_head(state),
            "value": self.value_head(state).squeeze(-1),
            "pair_logits": None,
        }
        if pair_candidates is not None:
            output["pair_logits"] = self.score_pair_candidates(
                state,
                pair_candidates,
                pair_features=pair_features,
            )
        return output

__all__ = [
    "EVENT_PAD",
    "EVENT_SINGLE",
    "EVENT_PAIR",
    "MonsterFeatureEncoder",
    "BPStateEncoder",
    "BPSharedNetwork",
]
