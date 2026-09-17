from __future__ import annotations

import json
from pathlib import Path

from bp_transformer_common import PROJECT_ROOT, normalize_name, slugify_fallback


DEFAULT_MECHANISM_PATH = PROJECT_ROOT / "json" / "sw_mechanism_profiles.json"

THREAT_LABELS = {
    "control": "硬控",
    "atb_control": "行动条控制",
    "cooldown_control": "技能冷却控制",
    "sustain": "续航",
    "revive": "复活",
    "bomb_dot": "炸弹/持续伤害",
    "defense": "防御增益",
    "tempo": "速度/回合循环",
}

THREAT_WEIGHTS = {
    "control": 1.25,
    "atb_control": 1.10,
    "cooldown_control": 0.90,
    "sustain": 1.00,
    "revive": 1.10,
    "bomb_dot": 1.00,
    "defense": 0.85,
    "tempo": 0.75,
}

THREAT_RESPONSES = {
    "control": ("cleanse", "immunity", "control_immunity"),
    "atb_control": ("atb_immunity", "tempo", "speed"),
    "cooldown_control": ("cooldown_reduce", "cooldown_immunity", "tempo"),
    "sustain": ("anti_heal", "anti_revive", "hp_destroy", "strip"),
    "revive": ("anti_revive", "oblivion", "hp_destroy"),
    "bomb_dot": ("cleanse", "immunity", "dot_immunity"),
    "defense": ("strip", "ignore_damage_reduction", "ignore_defense", "hp_destroy"),
    "tempo": ("atb_control", "speed_control", "cooldown_control"),
}


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class BPMechanismLayer:
    """Scores how much a candidate improves the current draft's mechanism coverage."""

    def __init__(self, profile_path=DEFAULT_MECHANISM_PATH):
        self.profile_path = Path(profile_path)
        data = _load_json(self.profile_path)
        raw_profiles = data.get("profiles_by_slug") if isinstance(data, dict) else {}
        self.profiles = {
            normalize_name(slug): profile
            for slug, profile in (raw_profiles or {}).items()
            if slug and isinstance(profile, dict)
        }
        self.alias_to_slug = {
            normalize_name(alias): normalize_name(slug)
            for alias, slug in (data.get("aliases_by_slug") or {}).items()
            if alias and slug
        }
        self.ready = bool(self.profiles)
        self.source_label = str(data.get("source") or "unavailable")
        self._resolve_cache = {}
        self._profile_cache = {}
        self._team_tags_cache = {}
        self._enemy_threats_cache = {}

    def _resolve_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        cached = self._resolve_cache.get(normalized)
        if cached is not None:
            return cached
        fallback = slugify_fallback(value)
        for key in (normalized, fallback):
            if key in self.profiles:
                self._resolve_cache[normalized] = key
                return key
            resolved = self.alias_to_slug.get(key)
            if resolved:
                self._resolve_cache[normalized] = resolved
                return resolved
        self._resolve_cache[normalized] = normalized
        return normalized

    def profile(self, value):
        normalized = normalize_name(value)
        if normalized not in self._profile_cache:
            self._profile_cache[normalized] = self.profiles.get(self._resolve_slug(value))
        return self._profile_cache[normalized]

    def _team_tags(self, values):
        key = tuple(self._resolve_slug(value) for value in values or [])
        cached = self._team_tags_cache.get(key)
        if cached is not None:
            tags, known = cached
            return set(tags), known
        tags = set()
        known = 0
        for slug in key:
            profile = self.profiles.get(slug)
            if not profile:
                continue
            known += 1
            tags.update(str(tag) for tag in profile.get("tags") or [])
        self._team_tags_cache[key] = (frozenset(tags), known)
        return tags, known

    @staticmethod
    def _coverage(tags, threat):
        responses = THREAT_RESPONSES.get(threat, ())
        return 1.0 if any(response in tags for response in responses) else 0.0

    def _enemy_threats(self, enemy_picks):
        key = tuple(self._resolve_slug(value) for value in enemy_picks or [])
        cached = self._enemy_threats_cache.get(key)
        if cached is not None:
            return cached
        threats = {}
        sources = {}
        known = 0
        for slug in key:
            profile = self.profiles.get(slug)
            if not profile:
                continue
            known += 1
            for threat in profile.get("threat_tags") or []:
                threat = str(threat)
                if threat not in THREAT_WEIGHTS:
                    continue
                threats[threat] = threats.get(threat, 0) + 1
                sources.setdefault(threat, []).append(str(profile.get("name") or value))

        weighted = {}
        for threat, count in threats.items():
            weighted[threat] = THREAT_WEIGHTS[threat] * min(1.5, 1.0 + 0.25 * (count - 1))
        result = (weighted, sources, known)
        self._enemy_threats_cache[key] = result
        return result

    def score_action(self, my_picks, enemy_picks, action):
        if not self.ready or not enemy_picks or not action:
            return {
                "available": False,
                "score": 0.5,
                "coverage": 0.0,
                "coverage_before": 0.0,
                "coverage_after": 0.0,
                "coverage_gain": 0.0,
                "known_enemy_count": 0,
                "threats": [],
                "action_tags": [],
                "reasons": [],
                "source": self.source_label,
            }

        threat_weights, sources, known_enemy_count = self._enemy_threats(enemy_picks)
        before_tags, known_my_count = self._team_tags(my_picks)
        action_tags, known_action_count = self._team_tags(action)
        after_tags = before_tags | action_tags
        if not threat_weights or not known_action_count:
            return {
                "available": bool(known_enemy_count),
                "score": 0.5,
                "coverage": 0.0,
                "coverage_before": 0.0,
                "coverage_after": 0.0,
                "coverage_gain": 0.0,
                "known_enemy_count": known_enemy_count,
                "known_my_count": known_my_count,
                "known_action_count": known_action_count,
                "threats": [],
                "action_tags": sorted(action_tags),
                "reasons": [],
                "source": self.source_label,
            }

        total_weight = sum(threat_weights.values())
        before_coverage = 0.0
        after_coverage = 0.0
        threat_rows = []
        reasons = []
        for threat, weight in sorted(threat_weights.items(), key=lambda item: item[1], reverse=True):
            before = self._coverage(before_tags, threat)
            after = self._coverage(after_tags, threat)
            before_coverage += weight * before
            after_coverage += weight * after
            gain = after - before
            responses = [
                response
                for response in THREAT_RESPONSES.get(threat, ())
                if response in action_tags
            ]
            row = {
                "threat": threat,
                "label": THREAT_LABELS.get(threat, threat),
                "weight": round(weight, 4),
                "before": before,
                "after": after,
                "gain": gain,
                "enemy_sources": sorted(set(sources.get(threat, []))),
                "responses": responses,
            }
            threat_rows.append(row)
            if gain > 0 and responses:
                reasons.append(
                    f"补足{row['label']}：{', '.join(responses)}"
                )

        before_coverage /= total_weight
        after_coverage /= total_weight
        coverage_gain = max(0.0, after_coverage - before_coverage)
        score = max(0.2, min(0.8, 0.5 + 0.30 * coverage_gain))
        return {
            "available": True,
            "score": round(score, 6),
            "coverage": round(after_coverage, 6),
            "coverage_before": round(before_coverage, 6),
            "coverage_after": round(after_coverage, 6),
            "coverage_gain": round(coverage_gain, 6),
            "known_enemy_count": known_enemy_count,
            "known_my_count": known_my_count,
            "known_action_count": known_action_count,
            "threats": threat_rows,
            "action_tags": sorted(action_tags),
            "reasons": reasons[:4],
            "source": self.source_label,
        }

    def score_ban(self, my_picks, enemy_picks, target):
        if not self.ready or not enemy_picks or not target:
            return {
                "available": False,
                "score": 0.5,
                "threat_reduction": 0.0,
                "known_enemy_count": 0,
                "known_target_count": 0,
                "threats": [],
                "reasons": [],
                "source": self.source_label,
            }

        target_slug = self._resolve_slug(target)
        before_weights, before_sources, known_enemy_count = self._enemy_threats(enemy_picks)
        remaining = [
            value
            for value in enemy_picks or []
            if self._resolve_slug(value) != target_slug
        ]
        after_weights, _after_sources, _known_remaining_count = self._enemy_threats(remaining)
        target_profile = self.profile(target)
        if not before_weights or not target_profile:
            return {
                "available": bool(known_enemy_count),
                "score": 0.5,
                "threat_reduction": 0.0,
                "known_enemy_count": known_enemy_count,
                "known_target_count": 1 if target_profile else 0,
                "threats": [],
                "reasons": [],
                "source": self.source_label,
            }

        total_weight = sum(before_weights.values())
        removed_weight = 0.0
        threat_rows = []
        reasons = []
        for threat, before_weight in sorted(before_weights.items(), key=lambda item: item[1], reverse=True):
            after_weight = after_weights.get(threat, 0.0)
            reduction = max(0.0, before_weight - after_weight)
            removed_weight += reduction
            row = {
                "threat": threat,
                "label": THREAT_LABELS.get(threat, threat),
                "before_weight": round(before_weight, 4),
                "after_weight": round(after_weight, 4),
                "reduction": round(reduction, 4),
                "enemy_sources": sorted(set(before_sources.get(threat, []))),
            }
            threat_rows.append(row)
            if reduction > 0:
                reasons.append(f"Ban 后削弱{row['label']}")

        threat_reduction = removed_weight / max(total_weight, 1e-6)
        score = max(0.2, min(0.8, 0.5 + 0.30 * threat_reduction))
        return {
            "available": True,
            "score": round(score, 6),
            "threat_reduction": round(threat_reduction, 6),
            "known_enemy_count": known_enemy_count,
            "known_target_count": 1,
            "threats": threat_rows,
            "reasons": reasons[:4],
            "source": self.source_label,
        }
