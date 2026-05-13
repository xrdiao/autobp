import torch
import torch.nn as nn

from bp_data_common import NameResolver
from bp_transformer_common import PAD_TOKEN, UNK_TOKEN, normalize_name


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
    first = "me" if normalize_name(first_pick_side) != "enemy" else "enemy"
    other = "enemy" if first == "me" else "me"
    pattern = [first, other, other, first, first, other, other, first, first, other]
    return pattern[order - 1]


def infer_pick_count(my_picks, enemy_picks, first_pick_side):
    next_order = len(my_picks or []) + len(enemy_picks or []) + 1
    count = 0
    while next_order + count <= 10 and side_for_order(next_order + count, first_pick_side) == "me":
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
        action_emb = self.encode_action(candidate_1, candidate_2, action_mask, action_type=action_type)
        features = torch.cat([state_emb, action_emb, state_emb * action_emb, torch.abs(state_emb - action_emb)], dim=-1)
        return self.scorer(features).squeeze(-1)


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
