from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

DATASET_ROOT = Path("dataset")
LEGACY_DATASET_VERSION = "legacy"
DEFAULT_VERSION_WEIGHT_MIN = 0.6
DEFAULT_VERSION_WEIGHT_MAX = 1.0
VERSION_DIR_RE = re.compile(r"^v(?P<body>\d+(?:\.\d+)*)$", re.IGNORECASE)


def normalize_dataset_version(value: object) -> str:
    if value is None:
        return LEGACY_DATASET_VERSION

    text = str(value).strip()
    if not text:
        return LEGACY_DATASET_VERSION

    match = VERSION_DIR_RE.match(text)
    if not match:
        return text.lower()

    parts = [str(int(part)) for part in match.group("body").split(".")]
    if len(parts) == 1:
        parts.append("0")
    return "v" + ".".join(parts)


def parse_dataset_version_parts(value: object) -> Optional[tuple[int, ...]]:
    normalized = normalize_dataset_version(value)
    match = VERSION_DIR_RE.match(normalized)
    if not match:
        return None
    return tuple(int(part) for part in match.group("body").split("."))


def is_version_dir_name(value: object) -> bool:
    return parse_dataset_version_parts(value) is not None


def dataset_version_sort_key(value: object) -> tuple[int, object]:
    normalized = normalize_dataset_version(value)
    if normalized == LEGACY_DATASET_VERSION:
        return (-2, ())

    parts = parse_dataset_version_parts(normalized)
    if parts is not None:
        return (0, parts)

    return (-1, normalized)


def sort_dataset_versions(values: Iterable[object]) -> list[str]:
    normalized = {normalize_dataset_version(value) for value in values if str(value or "").strip()}
    return sorted(normalized, key=dataset_version_sort_key)


def list_existing_dataset_versions(dataset_root: Path | str = DATASET_ROOT) -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        return []

    versions = []
    for child in root.iterdir():
        if child.is_dir() and is_version_dir_name(child.name):
            versions.append(normalize_dataset_version(child.name))
    return sort_dataset_versions(versions)


def auto_increment_dataset_version(existing_versions: Iterable[object]) -> str:
    numeric_versions = [parse_dataset_version_parts(value) for value in existing_versions]
    numeric_versions = [parts for parts in numeric_versions if parts is not None]
    if not numeric_versions:
        return "v1.0"

    latest = max(numeric_versions)
    return f"v{latest[0] + 1}.0"


def resolve_target_dataset_version(
    dataset_root: Path | str = DATASET_ROOT,
    requested_version: Optional[str] = None,
) -> str:
    env_version = os.getenv("BP_DATASET_VERSION", "").strip()
    if requested_version:
        return normalize_dataset_version(requested_version)
    if env_version:
        return normalize_dataset_version(env_version)
    return auto_increment_dataset_version(list_existing_dataset_versions(dataset_root))


def build_version_weight_map(
    versions: Iterable[object],
    min_weight: float = DEFAULT_VERSION_WEIGHT_MIN,
    max_weight: float = DEFAULT_VERSION_WEIGHT_MAX,
) -> dict[str, float]:
    ordered_versions = sort_dataset_versions(versions)
    if not ordered_versions:
        return {}

    if len(ordered_versions) == 1:
        return {ordered_versions[0]: float(max_weight)}

    span = float(max_weight) - float(min_weight)
    last_index = len(ordered_versions) - 1
    weights = {}
    for index, version in enumerate(ordered_versions):
        ratio = index / last_index
        weights[version] = round(float(min_weight) + span * ratio, 6)
    return weights


def get_dataset_version_weight(
    version: object,
    version_weights: Optional[dict[str, float]] = None,
    explicit_weight: object = None,
) -> float:
    if explicit_weight is not None:
        try:
            return float(explicit_weight)
        except Exception:
            pass

    normalized = normalize_dataset_version(version)
    if version_weights and normalized in version_weights:
        return float(version_weights[normalized])
    return 1.0


def extract_dataset_version_from_path(path: str | Path, dataset_root: Path | str = DATASET_ROOT) -> str:
    path_obj = Path(path)
    root = Path(dataset_root)

    try:
        rel_path = path_obj.resolve().relative_to(root.resolve())
    except Exception:
        rel_path = path_obj

    parts = rel_path.parts
    if parts and is_version_dir_name(parts[0]):
        return normalize_dataset_version(parts[0])
    return LEGACY_DATASET_VERSION


def summarize_dataset_version_counts(records) -> dict[str, int]:
    counter = Counter()
    for record in records or []:
        if isinstance(record, dict):
            counter[normalize_dataset_version(record.get("dataset_version"))] += 1

    return {
        version: counter.get(version, 0)
        for version in sort_dataset_versions(counter.keys())
    }


def describe_dataset_versions(versions: Iterable[object], version_weights: Optional[dict[str, float]] = None) -> list[dict[str, object]]:
    ordered_versions = sort_dataset_versions(versions)
    if version_weights is None:
        version_weights = build_version_weight_map(ordered_versions)

    latest_version = ordered_versions[-1] if ordered_versions else ""
    rows = []
    for index, version in enumerate(ordered_versions):
        rows.append(
            {
                "dataset_version": version,
                "version_order": index,
                "is_latest": version == latest_version,
                "version_weight": float(version_weights.get(version, 1.0)),
            }
        )
    return rows
