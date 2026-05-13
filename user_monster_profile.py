from copy import deepcopy


ACCOUNT_PROFILE_DEFAULT = {
    "profile_name": "",
    "contest_speed_1": None,
    "contest_speed_2": None,
    "fast_strip_speed": None,
    "followup_cc_speed": None,
    "followup_dps_speed": None,
    "turn2_support_speed": None,
    "speed_confidence": "B",
    "box_depth_speed": "B",
    "box_depth_bruiser": "B",
    "box_depth_anti_cc": "B",
    "box_depth_tech": "B",
    "notes": "",
}


MONSTER_BUILD_DEFAULT = {
    "monster": "",
    "build_state": 0,
    "actual_speed": None,
    "rune_tier": "C",
    "artifact_tier": "C",
    "skillup_complete": False,
    "confidence_score": 0.0,
    "preferred_build": "",
    "role_tags": [],
    "notes": "",
}


GRADE_OPTIONS = ["S", "A", "B", "C"]
BUILD_STATE_LABELS = {
    0: "Unavailable",
    1: "Partial",
    2: "Ready",
    3: "Core",
}
ROLE_TAG_OPTIONS = [
    "contest",
    "strip",
    "cc",
    "followup",
    "dps",
    "bruiser",
    "heal",
    "cleanse",
    "cutin",
    "anti_cc",
    "anti_cleave",
    "support",
    "lead",
]


def make_default_account_profile():
    return deepcopy(ACCOUNT_PROFILE_DEFAULT)


def make_default_monster_build(monster=""):
    data = deepcopy(MONSTER_BUILD_DEFAULT)
    data["monster"] = str(monster or "")
    return data


def _safe_int(value):
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value, default=0.0):
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_account_profile(data):
    merged = make_default_account_profile()
    if isinstance(data, dict):
        merged.update(data)
    for key in [
        "contest_speed_1",
        "contest_speed_2",
        "fast_strip_speed",
        "followup_cc_speed",
        "followup_dps_speed",
        "turn2_support_speed",
    ]:
        merged[key] = _safe_int(merged.get(key))
    for key in [
        "speed_confidence",
        "box_depth_speed",
        "box_depth_bruiser",
        "box_depth_anti_cc",
        "box_depth_tech",
    ]:
        value = str(merged.get(key) or "B").upper()
        merged[key] = value if value in GRADE_OPTIONS else "B"
    merged["profile_name"] = str(merged.get("profile_name") or "").strip()
    merged["notes"] = str(merged.get("notes") or "").strip()
    return merged


def normalize_monster_build_entry(monster, data):
    merged = make_default_monster_build(monster)
    if isinstance(data, dict):
        merged.update(data)
    merged["monster"] = str(monster or merged.get("monster") or "").strip()
    merged["build_state"] = max(0, min(3, int(_safe_int(merged.get("build_state")) or 0)))
    merged["actual_speed"] = _safe_int(merged.get("actual_speed"))
    merged["rune_tier"] = str(merged.get("rune_tier") or "C").upper()
    if merged["rune_tier"] not in GRADE_OPTIONS:
        merged["rune_tier"] = "C"
    merged["artifact_tier"] = str(merged.get("artifact_tier") or "C").upper()
    if merged["artifact_tier"] not in GRADE_OPTIONS:
        merged["artifact_tier"] = "C"
    merged["skillup_complete"] = bool(merged.get("skillup_complete", False))
    merged["confidence_score"] = max(0.0, min(1.0, _safe_float(merged.get("confidence_score"), default=0.0)))
    merged["preferred_build"] = str(merged.get("preferred_build") or "").strip()
    raw_tags = merged.get("role_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [item.strip() for item in raw_tags.split(",")]
    normalized_tags = []
    for tag in raw_tags:
        text = str(tag or "").strip().lower()
        if text and text not in normalized_tags:
            normalized_tags.append(text)
    merged["role_tags"] = normalized_tags
    merged["notes"] = str(merged.get("notes") or "").strip()
    return merged


def normalize_monster_builds(data):
    if not isinstance(data, dict):
        return {}
    result = {}
    for monster, item in data.items():
        normalized = normalize_monster_build_entry(monster, item)
        if normalized["build_state"] <= 0 and normalized["actual_speed"] is None and not normalized["preferred_build"] and not normalized["role_tags"] and not normalized["notes"] and not normalized["skillup_complete"] and normalized["confidence_score"] <= 0:
            continue
        result[normalized["monster"]] = normalized
    return dict(sorted(result.items(), key=lambda kv: kv[0].lower()))
