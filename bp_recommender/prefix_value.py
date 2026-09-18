import json

import torch

from bp_recommender.prefix_value_common import BPPrefixValueNet, encode_prefix_list
from bp_core.probability_calibration import apply_probability_calibration, resolve_probability_calibration
from bp_core.relation_common import MonsterVocab
from bp_core.transformer_common import PAD_TOKEN, PROJECT_ROOT, normalize_name, slugify_fallback

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_prefix_value_net.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_prefix_value_meta.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_prefix_value_vocab.json"


class BPPrefixValueRecommender:
    def __init__(
        self,
        model_path=str(DEFAULT_MODEL_PATH),
        meta_path=str(DEFAULT_META_PATH),
        vocab_path=str(DEFAULT_VOCAB_PATH),
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        self.vocab = MonsterVocab(vocab_dict)
        self.display_name_map = {
            normalize_name(key): str(value)
            for key, value in (self.meta.get("display_name_map") or {}).items()
            if key and value
        }
        self.image_name_map = {
            normalize_name(key): str(value)
            for key, value in (self.meta.get("image_name_map") or {}).items()
            if key and value
        }
        self.name_slug_aliases = {
            normalize_name(key): normalize_name(value)
            for key, value in (self.meta.get("name_slug_aliases") or {}).items()
            if key and value
        }
        for slug, display_name in self.display_name_map.items():
            self.name_slug_aliases.setdefault(slug, slug)
            self.name_slug_aliases.setdefault(normalize_name(display_name), slug)
            self.name_slug_aliases.setdefault(slugify_fallback(display_name), slug)

        self.prob_calibration = resolve_probability_calibration(self.meta, role="prefix_value")
        self.model = BPPrefixValueNet(
            vocab_size=len(self.vocab),
            embed_dim=int(self.meta.get("embed_dim", 128)),
            hidden_dim=int(self.meta.get("hidden_dim", 256)),
            dropout=float(self.meta.get("dropout", 0.15)),
            pad_idx=self.vocab.get_id(PAD_TOKEN),
            num_heads=int(self.meta.get("num_heads", 4)),
            num_layers=int(self.meta.get("num_layers", 3)),
            ffn_dim=int(self.meta.get("ffn_dim", 384)),
        )
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _to_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        fallback = slugify_fallback(value)
        for key in (normalized, fallback, fallback.replace("[", "").replace("]", "")):
            resolved = self.name_slug_aliases.get(key)
            if resolved:
                return normalize_name(resolved)
        return fallback or normalized

    def _to_display_name(self, slug):
        return self.display_name_map.get(normalize_name(slug), normalize_name(slug))

    @torch.inference_mode()
    def evaluate_states(self, states, return_detail=False):
        states = list(states or [])
        if not states:
            return []

        encoded_rows = []
        for state in states:
            my_slugs = [
                self._to_slug(value)
                for value in (state.get("my_picks") or [])
                if value
            ]
            enemy_slugs = [
                self._to_slug(value)
                for value in (state.get("enemy_picks") or [])
                if value
            ]
            my_ids, my_mask = encode_prefix_list(my_slugs, self.vocab)
            enemy_ids, enemy_mask = encode_prefix_list(enemy_slugs, self.vocab)
            next_turn = (
                len(my_slugs) + len(enemy_slugs) + 1
                if state.get("turn") is None
                else int(state["turn"])
            )
            first_pick_side = state.get("first_pick_side", "me")
            current_side = state.get("current_side", "me")
            first_pick_value = 1.0 if normalize_name(first_pick_side) == "me" else 0.0
            current_side_value = 1.0 if normalize_name(current_side) == "me" else 0.0
            my_leader_id = (
                self.vocab.get_id(self._to_slug(state["my_leader"]))
                if state.get("my_leader")
                else self.vocab.get_id(PAD_TOKEN)
            )
            enemy_leader_id = (
                self.vocab.get_id(self._to_slug(state["enemy_leader"]))
                if state.get("enemy_leader")
                else self.vocab.get_id(PAD_TOKEN)
            )
            encoded_rows.append(
                {
                    "my_ids": my_ids,
                    "my_mask": my_mask,
                    "enemy_ids": enemy_ids,
                    "enemy_mask": enemy_mask,
                    "my_leader_id": my_leader_id,
                    "enemy_leader_id": enemy_leader_id,
                    "turn": next_turn,
                    "first_pick_value": first_pick_value,
                    "current_side_value": current_side_value,
                }
            )

        pad_id = self.vocab.get_id(PAD_TOKEN)
        batch = {
            "my_picks": torch.tensor(
                [row["my_ids"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "my_mask": torch.tensor(
                [row["my_mask"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "enemy_picks": torch.tensor(
                [row["enemy_ids"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_mask": torch.tensor(
                [row["enemy_mask"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "my_ban": torch.full(
                (len(encoded_rows),),
                pad_id,
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_ban": torch.full(
                (len(encoded_rows),),
                pad_id,
                dtype=torch.long,
                device=self.device,
            ),
            "my_leader": torch.tensor(
                [row["my_leader_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_leader": torch.tensor(
                [row["enemy_leader_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "turn": torch.tensor(
                [float(row["turn"]) for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "first_pick_side": torch.tensor(
                [row["first_pick_value"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "current_side": torch.tensor(
                [row["current_side_value"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
        }
        raw_probs = torch.sigmoid(self.model(**batch)).detach().cpu().tolist()
        calibrated_probs = [
            apply_probability_calibration(raw_prob, self.prob_calibration)
            for raw_prob in raw_probs
        ]
        if not return_detail:
            return calibrated_probs

        results = []
        for state, row, raw_prob, calibrated_prob in zip(
            states,
            encoded_rows,
            raw_probs,
            calibrated_probs,
        ):
            results.append(
                {
                    "raw_win_rate": float(raw_prob),
                    "calibrated_win_rate": float(calibrated_prob),
                    "turn": row["turn"],
                    "current_side": (
                        "me" if row["current_side_value"] else "enemy"
                    ),
                    "model_type": self.meta.get(
                        "model_type",
                        "bp_prefix_value_model",
                    ),
                    "ban_features_used": False,
                }
            )
        return results

    @torch.inference_mode()
    def evaluate_state(
        self,
        my_picks,
        enemy_picks,
        turn=None,
        first_pick_side="me",
        current_side="me",
        my_ban=None,
        enemy_ban=None,
        my_leader=None,
        enemy_leader=None,
        return_detail=False,
    ):
        return self.evaluate_states(
            [
                {
                    "my_picks": my_picks,
                    "enemy_picks": enemy_picks,
                    "turn": turn,
                    "first_pick_side": first_pick_side,
                    "current_side": current_side,
                    "my_leader": my_leader,
                    "enemy_leader": enemy_leader,
                }
            ],
            return_detail=return_detail,
        )[0]
