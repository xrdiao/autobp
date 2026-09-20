import torch
import torch.nn as nn

from bp_core.data_common import NameResolver
from bp_core.transformer_common import PAD_TOKEN, UNK_TOKEN, normalize_name


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
        raw_slug = "" if slug is None else str(slug)
        normalized_slug = normalize_name(raw_slug)
        if not normalized_slug:
            return self.slug_to_id.get(UNK_TOKEN, 1)
        if normalized_slug == normalize_name(PAD_TOKEN):
            return self.slug_to_id.get(PAD_TOKEN, self.slug_to_id.get(normalized_slug, 0))
        if normalized_slug == normalize_name(UNK_TOKEN):
            return self.slug_to_id.get(UNK_TOKEN, self.slug_to_id.get(normalized_slug, 1))
        return self.slug_to_id.get(
            raw_slug,
            self.slug_to_id.get(
                normalized_slug,
                self.slug_to_id.get(UNK_TOKEN, self.slug_to_id.get(normalize_name(UNK_TOKEN), 1)),
            ),
        )

    def get_slug(self, idx):
        return self.id_to_slug.get(int(idx), UNK_TOKEN)

    def to_dict(self):
        return dict(self.slug_to_id)


def resolve_monster_slug(value, alias_map=None):
    return NameResolver(alias_map=alias_map).to_slug(value)


def normalize_slug_list(values, alias_map=None):
    slugs = []
    for value in values or []:
        slug = resolve_monster_slug(value, alias_map=alias_map)
        if slug:
            slugs.append(slug)
    return slugs


def side_for_order(order, first_pick_side):
    if order < 1 or order > 10:
        return None
    first = normalize_name(first_pick_side)
    if first not in {"me", "enemy"}:
        return None
    other = "enemy" if first == "me" else "me"
    pattern = [first, other, other, first, first, other, other, first, first, other]
    return pattern[order - 1]


def infer_pick_count(my_picks, enemy_picks, first_pick_side):
    return infer_pick_count_for_side(
        my_picks,
        enemy_picks,
        first_pick_side,
        "me",
    )


def infer_pick_count_for_side(my_picks, enemy_picks, first_pick_side, side):
    """Return how many consecutive picks belong to ``side`` at the current turn."""
    next_order = len(my_picks or []) + len(enemy_picks or []) + 1
    side = normalize_name(side)
    if side not in {"me", "enemy"}:
        return 0
    count = 0
    while (
        next_order + count <= 10
        and side_for_order(next_order + count, first_pick_side) == side
    ):
        count += 1
    return count


class BPDraftRelationNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        state_dim=256,
        action_dim=256,
        hidden_dim=256,
        dropout=0.15,
        pad_idx=0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = int(pad_idx)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.state_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
        )
        self.single_encoder = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
        )
        self.action_type_embedding = nn.Embedding(3, action_dim)
        nn.init.zeros_(self.action_type_embedding.weight)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _masked_mean(self, ids, mask):
        emb = self.embed(ids)
        mask = mask.to(dtype=emb.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom

    def encode_state(self, my_picks, my_mask, enemy_picks, enemy_mask, turn, first_pick_side):
        my_emb = self._masked_mean(my_picks, my_mask)
        enemy_emb = self._masked_mean(enemy_picks, enemy_mask)
        numeric = torch.stack([turn.float() / 10.0, first_pick_side.float()], dim=-1)
        return self.state_encoder(torch.cat([my_emb, enemy_emb, numeric], dim=-1))

    def encode_action(self, candidate_1, candidate_2, action_mask, action_type=None):
        emb_1 = self.embed(candidate_1)
        emb_2 = self.embed(candidate_2)
        pair_features = torch.cat([emb_1 + emb_2, emb_1 * emb_2, torch.abs(emb_1 - emb_2)], dim=-1)
        pair_emb = self.pair_encoder(pair_features)
        single_emb = self.single_encoder(emb_1)
        is_pair = (action_mask[:, 1] > 0.5).float().unsqueeze(-1)
        action_emb = single_emb * (1.0 - is_pair) + pair_emb * is_pair
        if action_type is None:
            action_type = (action_mask[:, 1] > 0.5).long()
        return action_emb + self.action_type_embedding(action_type.long().clamp(0, 2))

    def score_from_state(self, state_emb, candidate_1, candidate_2, action_mask, action_type=None):
        batch_size = candidate_1.size(0)
        if state_emb.size(0) == 1 and batch_size != 1:
            state_emb = state_emb.expand(batch_size, -1)
        action_emb = self.encode_action(candidate_1, candidate_2, action_mask, action_type=action_type)
        features = torch.cat([state_emb, action_emb, state_emb * action_emb, torch.abs(state_emb - action_emb)], dim=-1)
        return self.scorer(features).squeeze(-1)

    def forward(
        self,
        my_picks,
        my_mask,
        enemy_picks,
        enemy_mask,
        candidate_1,
        candidate_2,
        action_mask,
        turn,
        first_pick_side,
        action_type=None,
    ):
        state_emb = self.encode_state(my_picks, my_mask, enemy_picks, enemy_mask, turn, first_pick_side)
        return self.score_from_state(
            state_emb,
            candidate_1,
            candidate_2,
            action_mask,
            action_type=action_type,
        )


class BPDraftRelationNetV2(nn.Module):
    """Sequence-aware relation scorer for chronological BP prefixes."""

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        state_dim=256,
        action_dim=256,
        hidden_dim=256,
        dropout=0.15,
        pad_idx=0,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_idx = int(pad_idx)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(10, embed_dim)
        self.side_embedding = nn.Embedding(2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.state_output_norm = nn.LayerNorm(embed_dim)
        self.state_numeric = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
        )
        self.single_encoder = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
        )
        self.action_type_embedding = nn.Embedding(3, action_dim)
        nn.init.zeros_(self.action_type_embedding.weight)
        self.action_to_state = nn.MultiheadAttention(
            embed_dim=action_dim,
            num_heads=max(1, int(num_heads)),
            dropout=dropout,
            batch_first=True,
        )
        self.state_to_action = nn.Linear(embed_dim, action_dim)
        self.state_action_projection = nn.Linear(state_dim, action_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + action_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _state_tokens(
        self,
        my_picks,
        my_mask,
        enemy_picks,
        enemy_mask,
        first_pick_side,
    ):
        batch_size = my_picks.size(0)
        device = my_picks.device
        positions = torch.arange(10, device=device, dtype=torch.long).unsqueeze(0)
        me_positions = torch.tensor([0, 3, 4, 7, 8], dtype=torch.long, device=device)
        enemy_positions = torch.tensor([1, 2, 5, 6, 9], dtype=torch.long, device=device)

        chronological_me_first = torch.empty_like(torch.cat([my_picks, enemy_picks], dim=1))
        chronological_me_first_mask = torch.empty_like(torch.cat([my_mask, enemy_mask], dim=1))
        chronological_me_first[:, me_positions] = my_picks
        chronological_me_first[:, enemy_positions] = enemy_picks
        chronological_me_first_mask[:, me_positions] = my_mask
        chronological_me_first_mask[:, enemy_positions] = enemy_mask
        chronological_me_first_sides = torch.zeros((batch_size, 10), dtype=torch.long, device=device)
        chronological_me_first_sides[:, enemy_positions] = 1

        chronological_enemy_first = torch.empty_like(chronological_me_first)
        chronological_enemy_first_mask = torch.empty_like(chronological_me_first_mask)
        chronological_enemy_first[:, enemy_positions] = my_picks
        chronological_enemy_first[:, me_positions] = enemy_picks
        chronological_enemy_first_mask[:, enemy_positions] = my_mask
        chronological_enemy_first_mask[:, me_positions] = enemy_mask
        chronological_enemy_first_sides = torch.ones((batch_size, 10), dtype=torch.long, device=device)
        chronological_enemy_first_sides[:, enemy_positions] = 0

        first_is_me = first_pick_side.float().view(-1, 1) >= 0.5
        ids = torch.where(first_is_me, chronological_me_first, chronological_enemy_first)
        valid_mask = torch.where(
            first_is_me,
            chronological_me_first_mask,
            chronological_enemy_first_mask,
        ).bool()
        sides = torch.where(
            first_is_me,
            chronological_me_first_sides,
            chronological_enemy_first_sides,
        )
        tokens = (
            self.embed(ids)
            + self.position_embedding(positions)
            + self.side_embedding(sides)
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = self.dropout(self.input_norm(sequence))
        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
                ~valid_mask,
            ],
            dim=1,
        )
        encoded = self.state_transformer(sequence, src_key_padding_mask=padding_mask)
        return self.state_output_norm(encoded), padding_mask

    def encode_state(
        self,
        my_picks,
        my_mask,
        enemy_picks,
        enemy_mask,
        turn,
        first_pick_side,
        current_side=None,
    ):
        state_tokens, padding_mask = self._state_tokens(
            my_picks,
            my_mask,
            enemy_picks,
            enemy_mask,
            first_pick_side,
        )
        if current_side is None:
            current_side = torch.ones_like(first_pick_side)
        numeric = torch.stack(
            [
                turn.float() / 10.0,
                first_pick_side.float(),
                current_side.float(),
            ],
            dim=-1,
        )
        numeric_emb = self.state_numeric(numeric)
        state_emb = self.state_encoder(torch.cat([state_tokens[:, 0], numeric_emb], dim=-1))
        return state_emb, state_tokens, padding_mask

    def encode_action(self, candidate_1, candidate_2, action_mask, action_type=None):
        emb_1 = self.embed(candidate_1)
        emb_2 = self.embed(candidate_2)
        pair_features = torch.cat(
            [emb_1 + emb_2, emb_1 * emb_2, torch.abs(emb_1 - emb_2)],
            dim=-1,
        )
        pair_emb = self.pair_encoder(pair_features)
        single_emb = self.single_encoder(emb_1)
        is_pair = (action_mask[:, 1] > 0.5).float().unsqueeze(-1)
        action_emb = single_emb * (1.0 - is_pair) + pair_emb * is_pair
        if action_type is None:
            action_type = (action_mask[:, 1] > 0.5).long()
        return action_emb + self.action_type_embedding(action_type.long().clamp(0, 2))

    def score_from_state(self, state_cache, candidate_1, candidate_2, action_mask, action_type=None):
        state_emb, state_tokens, padding_mask = state_cache
        batch_size = candidate_1.size(0)
        if state_emb.size(0) == 1 and batch_size != 1:
            state_emb = state_emb.expand(batch_size, -1)
            state_tokens = state_tokens.expand(batch_size, -1, -1)
            padding_mask = padding_mask.expand(batch_size, -1)
        action_emb = self.encode_action(candidate_1, candidate_2, action_mask, action_type=action_type)
        action_query = action_emb.unsqueeze(1)
        action_keys = self.state_to_action(state_tokens)
        cross_context, _ = self.action_to_state(
            query=action_query,
            key=action_keys,
            value=action_keys,
            key_padding_mask=padding_mask,
        )
        cross_context = cross_context.squeeze(1)
        state_action = self.state_action_projection(state_emb)
        features = torch.cat(
            [
                state_emb,
                action_emb,
                cross_context,
                state_action * cross_context,
                torch.abs(state_action - cross_context),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)

    def forward(
        self,
        my_picks,
        my_mask,
        enemy_picks,
        enemy_mask,
        candidate_1,
        candidate_2,
        action_mask,
        turn,
        first_pick_side,
        action_type=None,
        current_side=None,
    ):
        state_emb, state_tokens, padding_mask = self.encode_state(
            my_picks=my_picks,
            my_mask=my_mask,
            enemy_picks=enemy_picks,
            enemy_mask=enemy_mask,
            turn=turn,
            first_pick_side=first_pick_side,
            current_side=current_side,
        )
        return self.score_from_state(
            (state_emb, state_tokens, padding_mask),
            candidate_1,
            candidate_2,
            action_mask,
            action_type=action_type,
        )


def encode_pick_list(values, vocab, to_slug, max_len=5):
    ids = [vocab.get_id(to_slug(value)) for value in (values or [])[:max_len]]
    mask = [1.0] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab.get_id(PAD_TOKEN))
        mask.append(0.0)
    return ids, mask


def valid_vocab_slugs(vocab):
    blocked = {normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN)}
    return [vocab.get_slug(i) for i in range(len(vocab)) if normalize_name(vocab.get_slug(i)) not in blocked]
