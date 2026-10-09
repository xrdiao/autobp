"""Shrink a value head until its spread clears the model's own residual.

A fixed cutoff would flip decisions on either side of an arbitrary line.
The weight below is zero while this shortlist is still inside the residual
noise, then rises smoothly to one as the candidates separate further.
"""

from __future__ import annotations


def _percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, float(quantile))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def robust_spread(values):
    """Return the 90th-minus-10th percentile spread of one decision's values."""
    values = list(values or [])
    if len(values) < 2:
        return 0.0
    return _percentile(values, 0.9) - _percentile(values, 0.1)


def median_value(values):
    ordered = sorted(float(value) for value in values or [])
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def influence_weight(values, noise):
    """Map shortlist spread and residual noise onto a weight in ``[0, 1]``."""
    if len(list(values or [])) < 2 or noise is None:
        return 0.0
    spread = robust_spread(values)
    noise = float(noise)
    if noise <= 0.0:
        return 1.0 if spread > 0.0 else 0.0
    return max(0.0, min(1.0, (spread - noise) / noise))


def value_adjustments(values, noise):
    """Return ``(weight, median, per-value adjustment)`` for one shortlist."""
    values = [float(value) for value in values or []]
    weight = influence_weight(values, noise)
    center = median_value(values)
    return weight, center, [weight * (value - center) for value in values]


def resolve_value_noise(meta):
    """Read a residual measured at training time.

    Missing measurements leave the value weight at zero. A hand-picked
    fallback would start reordering before the residual has been measured.
    """
    if not isinstance(meta, dict) or meta.get("value_residual_noise") is None:
        return None
    return float(meta["value_residual_noise"])


def median_absolute_residual(probabilities, labels, weights=None):
    """Weighted median of ``|prediction - label|`` on a validation split."""
    residuals = [abs(float(probability) - float(label)) for probability, label in zip(probabilities, labels)]
    if not residuals:
        return 0.0
    if weights is None:
        weights = [1.0] * len(residuals)
    order = sorted(range(len(residuals)), key=lambda index: residuals[index])
    total = sum(max(0.0, float(weights[index])) for index in order)
    if total <= 0.0:
        return residuals[order[len(order) // 2]]
    covered = 0.0
    halfway = total / 2.0
    for index in order:
        covered += max(0.0, float(weights[index]))
        if covered >= halfway:
            return residuals[index]
    return residuals[order[-1]]
