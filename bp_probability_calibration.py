import math


_ROLE_DEFAULTS = {
    "policy_value": {
        "transformer": {"temperature_floor": 1.35, "prior_floor": 0.02},
        "legacy": {"temperature_floor": 1.0, "prior_floor": 0.0},
    },
    "partial_value": {
        "transformer": {"temperature_floor": 4.0, "prior_floor": 0.10},
        "legacy": {"temperature_floor": 1.0, "prior_floor": 0.0},
    },
    "draft_value": {
        "transformer": {"temperature_floor": 5.5, "prior_floor": 0.10},
        "legacy": {"temperature_floor": 4.0, "prior_floor": 0.0},
    },
}


def _clip_probability(prob, clip_min=1e-4, clip_max=1.0 - 1e-4):
    return min(float(clip_max), max(float(clip_min), float(prob)))


def resolve_probability_calibration(meta, role):
    role_defaults = _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["policy_value"])
    model_type = str(meta.get("model_type", "")).lower()
    is_transformer = "transformer" in model_type or meta.get("architecture") == "transformer_encoder"
    family_defaults = role_defaults["transformer" if is_transformer else "legacy"]

    clip_min = float(meta.get("prob_clip_min", 1e-4))
    clip_max = float(meta.get("prob_clip_max", 1.0 - 1e-4))

    raw_temperature = float(meta.get("prob_temperature", 1.0))
    raw_prior = float(meta.get("prob_prior_weight", 0.0))

    return {
        "clip_min": clip_min,
        "clip_max": clip_max,
        "temperature": max(raw_temperature, family_defaults["temperature_floor"]),
        "prior_weight": max(raw_prior, family_defaults["prior_floor"]),
    }


def apply_probability_calibration(raw_prob, calibration):
    prob = _clip_probability(
        raw_prob,
        clip_min=calibration.get("clip_min", 1e-4),
        clip_max=calibration.get("clip_max", 1.0 - 1e-4),
    )

    temperature = float(calibration.get("temperature", 1.0))
    if temperature and temperature != 1.0:
        logit = math.log(prob / (1.0 - prob))
        prob = 1.0 / (1.0 + math.exp(-(logit / temperature)))

    prior_weight = min(max(float(calibration.get("prior_weight", 0.0)), 0.0), 1.0)
    if prior_weight > 0.0:
        prob = (1.0 - prior_weight) * prob + prior_weight * 0.5

    return _clip_probability(
        prob,
        clip_min=calibration.get("clip_min", 1e-4),
        clip_max=calibration.get("clip_max", 1.0 - 1e-4),
    )
