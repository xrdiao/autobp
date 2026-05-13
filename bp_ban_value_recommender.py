import json

import torch

from bp_probability_calibration import apply_probability_calibration, resolve_probability_calibration
from bp_transformer_common import PAD_TOKEN, PROJECT_ROOT, UNK_TOKEN, normalize_name, slugify_fallback
from train_draft_value_net import BPDraftValueNet as TransformerDraftValueNet

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_draft_value_net.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_draft_value_meta.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_draft_value_vocab.json"


class MonsterVocab:
    def __init__(self, slug_to_id):
        self.slug_to_id = dict(slug_to_id)
        self.id_to_slug = {v: k for k, v in self.slug_to_id.items()}

    def __len__(self):
        return len(self.slug_to_id)

    def get_id(self, slug):
        slug = normalize_name(slug)
        return self.slug_to_id.get(slug, self.slug_to_id.get(UNK_TOKEN, 1))


class BPBanValueRecommender:
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
        self.display_name_map = self.meta.get("display_name_map", {})
        self.image_name_map = self.meta.get("image_name_map", {})
        self.prob_calibration = resolve_probability_calibration(self.meta, role="draft_value")
        self.prob_temperature = float(self.prob_calibration["temperature"])
        self.prob_prior_weight = float(self.prob_calibration["prior_weight"])
        self.prob_clip_min = float(self.prob_calibration["clip_min"])
        self.prob_clip_max = float(self.prob_calibration["clip_max"])

        model_type = str(self.meta.get("model_type", "")).lower()
        if "transformer" in model_type or self.meta.get("architecture") == "transformer_encoder":
            model_cls = TransformerDraftValueNet
            model_kwargs = {
                "vocab_size": len(self.vocab),
                "embed_dim": self.meta["embed_dim"],
                "hidden_dim": self.meta["hidden_dim"],
                "dropout": self.meta["dropout"],
                "pad_idx": self.vocab.get_id(PAD_TOKEN),
                "num_heads": int(self.meta.get("num_heads", 4)),
                "num_layers": int(self.meta.get("num_layers", 3)),
                "ffn_dim": int(self.meta.get("ffn_dim", max(self.meta["hidden_dim"], self.meta["embed_dim"] * 3))),
            }
        else:
            raise RuntimeError(
                f"Unsupported draft value model type: {self.meta.get('model_type')}. "
                "Retrain with train_draft_value_net.py to produce the current transformer model."
            )

        self.model = model_cls(**model_kwargs)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _to_slug(self, value):
        return slugify_fallback(value)

    def _to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug, slug)

    def _to_image_name(self, slug):
        slug = normalize_name(slug)
        return self.image_name_map.get(slug, self._to_display_name(slug))

    def _encode_list(self, arr, max_len=5):
        ids = [self.vocab.get_id(self._to_slug(x)) for x in (arr or [])[:max_len]]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.vocab.get_id(PAD_TOKEN))
            mask.append(0)
        return ids, mask

    @torch.no_grad()
    def evaluate_state(
        self,
        my_picks,
        enemy_picks,
        my_ban=None,
        enemy_ban=None,
        my_leader=None,
        enemy_leader=None,
        return_detail=False,
    ):
        my_ids, my_mask = self._encode_list(my_picks, 5)
        enemy_ids, enemy_mask = self._encode_list(enemy_picks, 5)

        my_ban_id = self.vocab.get_id(self._to_slug(my_ban)) if my_ban else self.vocab.get_id(PAD_TOKEN)
        enemy_ban_id = self.vocab.get_id(self._to_slug(enemy_ban)) if enemy_ban else self.vocab.get_id(PAD_TOKEN)
        my_leader_id = self.vocab.get_id(self._to_slug(my_leader)) if my_leader else self.vocab.get_id(PAD_TOKEN)
        enemy_leader_id = self.vocab.get_id(self._to_slug(enemy_leader)) if enemy_leader else self.vocab.get_id(PAD_TOKEN)

        batch = {
            "my_picks": torch.tensor([my_ids], dtype=torch.long, device=self.device),
            "my_mask": torch.tensor([my_mask], dtype=torch.float32, device=self.device),
            "enemy_picks": torch.tensor([enemy_ids], dtype=torch.long, device=self.device),
            "enemy_mask": torch.tensor([enemy_mask], dtype=torch.float32, device=self.device),
            "my_ban": torch.tensor([my_ban_id], dtype=torch.long, device=self.device),
            "enemy_ban": torch.tensor([enemy_ban_id], dtype=torch.long, device=self.device),
            "my_leader": torch.tensor([my_leader_id], dtype=torch.long, device=self.device),
            "enemy_leader": torch.tensor([enemy_leader_id], dtype=torch.long, device=self.device),
        }

        logits = self.model(**batch)
        raw_prob = torch.sigmoid(logits)[0].item()
        calibrated_prob = apply_probability_calibration(raw_prob, self.prob_calibration)

        if return_detail:
            return {
                "raw_win_rate": raw_prob,
                "calibrated_win_rate": calibrated_prob,
            }
        return calibrated_prob

    @torch.no_grad()
    def recommend_bans(
        self,
        my_picks,
        enemy_picks,
        my_leader=None,
        enemy_leader=None,
        enemy_ban=None,
        top_k=5,
    ):
        candidates = [self._to_slug(x) for x in (enemy_picks or []) if x]
        candidates = list(dict.fromkeys(candidates))

        base_eval = self.evaluate_state(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            my_ban=None,
            enemy_ban=enemy_ban,
            my_leader=my_leader,
            enemy_leader=enemy_leader,
            return_detail=True,
        )
        base_wr = float(base_eval["calibrated_win_rate"])
        base_raw_wr = float(base_eval["raw_win_rate"])

        plans = []
        for target in candidates:
            eval_result = self.evaluate_state(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                my_ban=target,
                enemy_ban=enemy_ban,
                my_leader=my_leader,
                enemy_leader=enemy_leader,
                return_detail=True,
            )
            wr = float(eval_result["calibrated_win_rate"])
            raw_wr = float(eval_result["raw_win_rate"])
            gain = wr - base_wr
            raw_gain = raw_wr - base_raw_wr

            display = self._to_display_name(target)
            image_name = self._to_image_name(target)

            plans.append(
                {
                    "ban": display,
                    "ban_slug": target,
                    "plan": [display],
                    "plan_slugs": [target],
                    "image_names": [image_name],
                    "score": round(wr, 6),
                    "plan_score": round(wr, 6),
                    "estimated_win_rate": round(wr, 6),
                    "source": str(self.meta.get("model_type", "bp_draft_value_model")),
                    "summary": "Ban recommendation from full-draft value model",
                    "reasons": [
                        "Compares full draft win rate before and after banning the target",
                        "Ranks bans by post-ban expected win rate",
                    ],
                    "detail": {
                        "ban_target": display,
                        "pre_ban_win_rate": round(base_wr, 6),
                        "post_ban_win_rate": round(wr, 6),
                        "win_rate_gain": round(gain, 6),
                        "raw_pre_ban_win_rate": round(base_raw_wr, 6),
                        "raw_post_ban_win_rate": round(raw_wr, 6),
                        "raw_win_rate_gain": round(raw_gain, 6),
                        "prob_temperature": round(self.prob_temperature, 6),
                        "prob_prior_weight": round(self.prob_prior_weight, 6),
                    },
                    "monster": display,
                }
            )

        plans.sort(key=lambda x: x["plan_score"], reverse=True)
        return plans[:top_k]
