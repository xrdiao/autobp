"""The account input used by AutoBP: an explicit list of owned monsters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bp_core.names import normalize_name


@dataclass(frozen=True)
class AccountProfile:
    """Normalized ownership data used only for the optional Box mask."""

    owned_monsters: frozenset[str] = frozenset()
    source: str = "manual"
    source_version: str = ""

    @classmethod
    def from_payload(cls, payload: dict) -> "AccountProfile":
        if not isinstance(payload, dict):
            raise TypeError("account profile payload must be an object")
        values = payload.get("owned_monsters")
        if not isinstance(values, (list, tuple, set)):
            raise ValueError("account profile requires an owned_monsters list")
        return cls(
            owned_monsters=frozenset(
                normalized
                for value in values
                if (normalized := normalize_name(value))
            ),
            source=str(payload.get("source") or "manual"),
            source_version=str(payload.get("source_version") or ""),
        )


def load_account_profile(path: Path) -> AccountProfile | None:
    """Load the optional ownership file; absent files mean no Box mask."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return AccountProfile.from_payload(json.load(file))
