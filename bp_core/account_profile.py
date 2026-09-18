"""Normalized account and rune profile data used by the BP decision layer.

The profile deliberately does not guess the layout of an exported SWEX file.
An importer can translate a verified export into this small schema later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bp_core.transformer_common import normalize_name


RUNE_FIELDS = (
    "speed",
    "hp",
    "atk",
    "def",
    "crit_rate",
    "crit_damage",
    "accuracy",
    "resistance",
    "sets",
    "rune_ids",
    "rune_set_ids",
    "rune_slots",
)


@dataclass(frozen=True)
class RuneInventoryEntry:
    rune_id: str
    slot: int | None = None
    set_id: str = ""
    primary_stat: tuple[float, ...] = ()
    secondary_stats: tuple[tuple[float, ...], ...] = ()
    equipped_monster: str = ""


@dataclass(frozen=True)
class RuneProfile:
    monster_slug: str
    speed: float | None = None
    hp: float | None = None
    atk: float | None = None
    defense: float | None = None
    crit_rate: float | None = None
    crit_damage: float | None = None
    accuracy: float | None = None
    resistance: float | None = None
    sets: tuple[str, ...] = ()
    rune_ids: tuple[str, ...] = ()
    rune_set_ids: tuple[str, ...] = ()
    rune_slots: tuple[int, ...] = ()
    verified_fields: tuple[str, ...] = ()
    confidence: float = 0.0
    source: str = "unknown"

    @property
    def has_verified_data(self) -> bool:
        return bool(self.verified_fields)


@dataclass
class AccountProfile:
    owned_monsters: frozenset[str] = frozenset()
    runes_by_monster: dict[str, RuneProfile] = field(default_factory=dict)
    rune_inventory: tuple[RuneInventoryEntry, ...] = ()
    source: str = "unknown"
    source_version: str = ""
    parse_confidence: float = 0.0
    unparsed_fields: tuple[str, ...] = ()

    @property
    def has_verified_runes(self) -> bool:
        return bool(self.rune_inventory) or any(
            profile.has_verified_data for profile in self.runes_by_monster.values()
        )

    @property
    def rune_inventory_count(self) -> int:
        return len(self.rune_inventory)

    @classmethod
    def from_payload(cls, payload: dict) -> "AccountProfile":
        if not isinstance(payload, dict):
            raise TypeError("account profile payload must be an object")

        owned = frozenset(
            normalize_name(value)
            for value in payload.get("owned_monsters", [])
            if normalize_name(value)
        )
        rune_profiles = {}
        for raw_name, raw_profile in (payload.get("runes_by_monster", {}) or {}).items():
            if not isinstance(raw_profile, dict):
                continue
            monster_slug = normalize_name(raw_name)
            if not monster_slug:
                continue
            verified_fields = tuple(
                field_name
                for field_name in raw_profile.get("verified_fields", [])
                if field_name in RUNE_FIELDS
            )
            rune_profiles[monster_slug] = RuneProfile(
                monster_slug=monster_slug,
                speed=raw_profile.get("speed"),
                hp=raw_profile.get("hp"),
                atk=raw_profile.get("atk"),
                defense=raw_profile.get("def"),
                crit_rate=raw_profile.get("crit_rate"),
                crit_damage=raw_profile.get("crit_damage"),
                accuracy=raw_profile.get("accuracy"),
                resistance=raw_profile.get("resistance"),
                sets=tuple(str(value) for value in raw_profile.get("sets", []) if str(value).strip()),
                rune_ids=tuple(str(value) for value in raw_profile.get("rune_ids", []) if str(value).strip()),
                rune_set_ids=tuple(str(value) for value in raw_profile.get("rune_set_ids", []) if str(value).strip()),
                rune_slots=tuple(int(value) for value in raw_profile.get("rune_slots", []) if str(value).strip()),
                verified_fields=verified_fields,
                confidence=float(raw_profile.get("confidence", 0.0) or 0.0),
                source=str(raw_profile.get("source", payload.get("source", "unknown"))),
            )

        known_fields = {
            "schema_version",
            "source",
            "source_version",
            "parse_confidence",
            "owned_monsters",
            "runes_by_monster",
            "rune_inventory",
        }
        return cls(
            owned_monsters=owned,
            runes_by_monster=rune_profiles,
            rune_inventory=tuple(
                RuneInventoryEntry(
                    rune_id=str(value.get("rune_id", "")),
                    slot=int(value["slot"]) if value.get("slot") is not None else None,
                    set_id=str(value.get("set_id", "")),
                    primary_stat=tuple(float(item) for item in value.get("primary_stat", [])),
                    secondary_stats=tuple(
                        tuple(float(item) for item in stat)
                        for stat in value.get("secondary_stats", [])
                    ),
                    equipped_monster=normalize_name(value.get("equipped_monster")),
                )
                for value in payload.get("rune_inventory", [])
                if isinstance(value, dict) and value.get("rune_id") is not None
            ),
            source=str(payload.get("source", "unknown")),
            source_version=str(payload.get("source_version", "")),
            parse_confidence=float(payload.get("parse_confidence", 0.0) or 0.0),
            unparsed_fields=tuple(sorted(set(payload) - known_fields)),
        )


def load_account_profile(path: Path) -> AccountProfile | None:
    """Load normalized JSON or a raw SWEX profile, if present."""
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "unit_list" in payload:
        from bp_core.swex_profile import DEFAULT_CATALOG_PATH, parse_swex_payload

        with open(DEFAULT_CATALOG_PATH, "r", encoding="utf-8") as file:
            catalog = json.load(file)
        return parse_swex_payload(payload, catalog=catalog)
    return AccountProfile.from_payload(payload)
