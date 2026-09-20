"""Translate the stable, verified parts of a SWEX profile into BP inputs."""

from __future__ import annotations

import json

from bp_core.account_profile import AccountProfile, RuneInventoryEntry, RuneProfile
from bp_core.transformer_common import PROJECT_ROOT, normalize_name


DEFAULT_CATALOG_PATH = PROJECT_ROOT / "json" / "lussack_monster_catalog.json"


def _number(value):
    if value is None or value == "":
        return None
    return float(value)


def _catalog_map(catalog):
    result = {}
    for item in catalog or []:
        unit_id = str(item.get("com2us_id") or "").strip()
        slug = normalize_name(item.get("slug"))
        if unit_id and slug:
            result[unit_id] = {
                "slug": slug,
                "name": str(item.get("slabel") or item.get("label") or slug),
            }
    return result


def _stat_tuple(value):
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(float(item) for item in value)


def _rune_inventory_entry(rune, equipped_monster=""):
    return RuneInventoryEntry(
        rune_id=str(rune.get("rune_id")),
        slot=int(rune["slot_no"]) if rune.get("slot_no") is not None else None,
        set_id=str(rune.get("set_id") or ""),
        primary_stat=_stat_tuple(rune.get("pri_eff")),
        secondary_stats=tuple(
            _stat_tuple(stat)
            for stat in rune.get("sec_eff") or []
            if isinstance(stat, (list, tuple))
        ),
        equipped_monster=normalize_name(equipped_monster),
    )


def parse_swex_payload(payload, catalog=None):
    """Parse ownership and equipped-unit build data from SWEX.

    The parser intentionally uses only unit_master_id, unit stats, rune_id,
    slot_no, and set_id. RTA-specific equipment pools are not inferred from a
    generic export because the export can contain non-RTA equipment states.
    """
    if not isinstance(payload, dict):
        raise TypeError("SWEX payload must be an object")

    catalog_by_id = _catalog_map(catalog)
    owned = set()
    rune_profiles = {}
    unresolved = []
    rune_inventory = {}
    for rune in payload.get("runes") or []:
        if isinstance(rune, dict) and rune.get("rune_id") is not None:
            entry = _rune_inventory_entry(rune)
            rune_inventory[entry.rune_id] = entry
    units = payload.get("unit_list") or []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_master_id") or "").strip()
        catalog_item = catalog_by_id.get(unit_id)
        if catalog_item is None:
            if unit_id:
                unresolved.append(unit_id)
            continue

        slug = catalog_item["slug"]
        owned.add(slug)
        equipped_runes = [rune for rune in unit.get("runes") or [] if isinstance(rune, dict)]
        for rune in equipped_runes:
            if rune.get("rune_id") is not None:
                entry = _rune_inventory_entry(rune, slug)
                rune_inventory[entry.rune_id] = entry
        rune_ids = tuple(
            str(rune.get("rune_id"))
            for rune in equipped_runes
            if rune.get("rune_id") is not None
        )
        rune_set_ids = tuple(
            str(rune.get("set_id"))
            for rune in equipped_runes
            if rune.get("set_id") is not None
        )
        rune_slots = tuple(
            int(rune.get("slot_no"))
            for rune in equipped_runes
            if rune.get("slot_no") is not None
        )
        verified_fields = ["speed"] if unit.get("spd") is not None else []
        if rune_set_ids:
            verified_fields.append("rune_set_ids")
        if rune_ids:
            verified_fields.append("rune_ids")
        if rune_slots:
            verified_fields.append("rune_slots")
        rune_profiles[slug] = RuneProfile(
            monster_slug=slug,
            speed=_number(unit.get("spd")),
            hp=_number(unit.get("con")),
            atk=_number(unit.get("atk")),
            defense=_number(unit.get("def")),
            crit_rate=_number(unit.get("critical_rate")),
            crit_damage=_number(unit.get("critical_damage")),
            accuracy=_number(unit.get("accuracy")),
            resistance=_number(unit.get("resist")),
            sets=(),
            rune_ids=rune_ids,
            rune_set_ids=rune_set_ids,
            rune_slots=rune_slots,
            verified_fields=tuple(verified_fields),
            confidence=1.0,
            source="swex",
        )

    parsed_count = len(rune_profiles)
    total_units = sum(1 for unit in units if isinstance(unit, dict) and unit.get("unit_master_id"))
    parse_confidence = parsed_count / total_units if total_units else 0.0
    return AccountProfile(
        owned_monsters=frozenset(owned),
        runes_by_monster=rune_profiles,
        rune_inventory=tuple(rune_inventory.values()),
        source="swex",
        source_version=str(payload.get("version") or payload.get("swex_version") or ""),
        parse_confidence=round(parse_confidence, 6),
        unparsed_fields=tuple(
            ["rta_equipment_pool"]
            + [f"unit_master_id:{unit_id}" for unit_id in sorted(set(unresolved))]
        ),
    )


def load_swex_profile(path, catalog_path=DEFAULT_CATALOG_PATH):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    with open(catalog_path, "r", encoding="utf-8") as file:
        catalog = json.load(file)
    return parse_swex_payload(payload, catalog=catalog)
