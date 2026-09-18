"""Own-account feasibility signals for BP recommendations.

Rune data is useful as a constraint for the user's account. It is not a
prediction of the opponent's hidden build and is intentionally absent from
the adversarial score for now.
"""

from __future__ import annotations

from dataclasses import dataclass

from bp_core.account_profile import AccountProfile
from bp_core.transformer_common import normalize_name


@dataclass(frozen=True)
class FeasibilityResult:
    score: float
    owned: bool | None
    verified_rune_data: bool
    confidence: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "owned": self.owned,
            "verified_rune_data": self.verified_rune_data,
            "confidence": round(self.confidence, 6),
            "reasons": list(self.reasons),
        }


def plan_monsters(plan) -> list[str]:
    if isinstance(plan, dict):
        values = plan.get("plan", plan.get("monsters", []))
    else:
        values = plan
    return [normalize_name(value) for value in values or [] if normalize_name(value)]


def score_plan_feasibility(plan, account_profile: AccountProfile | None) -> FeasibilityResult:
    """Return an explainable own-account signal without changing BP ranking."""
    if account_profile is None:
        return FeasibilityResult(
            score=0.5,
            owned=None,
            verified_rune_data=False,
            confidence=0.0,
            reasons=("未加载账号档案，未使用符文约束",),
        )

    monsters = plan_monsters(plan)
    owned = None
    reasons = []
    if account_profile.owned_monsters:
        missing = [monster for monster in monsters if monster not in account_profile.owned_monsters]
        owned = not missing
        if missing:
            reasons.append("推荐包含账号 Box 中未确认拥有的魔灵")
        else:
            reasons.append("推荐中的魔灵均在账号档案中")

    rune_profiles = [
        account_profile.runes_by_monster.get(monster)
        for monster in monsters
    ]
    verified_rune_data = bool(rune_profiles) and all(
        profile is not None and profile.has_verified_data
        for profile in rune_profiles
    )
    if verified_rune_data:
        reasons.append("推荐中的魔灵均有已验证符文字段")
    elif monsters:
        reasons.append("符文字段不完整，仅作为参考")

    score = 0.5 if owned is None else (1.0 if owned else 0.0)
    return FeasibilityResult(
        score=score,
        owned=owned,
        verified_rune_data=verified_rune_data,
        confidence=float(account_profile.parse_confidence),
        reasons=tuple(reasons),
    )
