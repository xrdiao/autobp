from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = PROJECT_ROOT / "json"
CATALOG_PATH = JSON_DIR / "monster_catalog.json"
LUCK_CATALOG_PATH = JSON_DIR / "lucksack_monster_catalog.json"
SWKIT_ALIAS_PATH = JSON_DIR / "swkit_monster_aliases.json"


def normalize_standard_name(name: str) -> str:
    return str(name or "").strip()


def _load_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def _first_value(value):
    if isinstance(value, (list, tuple)):
        return next((str(item).strip() for item in value if str(item).strip()), "")
    return str(value or "").strip()


def _lucksack_image_url(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return f"https://static.lucksack.gg/images/monsters/{raw.lstrip('/')}"


def make_aliases_for_entry(entry):
    aliases = set()
    for key in ("base_name", "name", "label", "slabel", "slug", "com2us_id"):
        value = entry.get(key)
        if value:
            aliases.add(str(value))

    for key in ("raw_names", "alts", "titles", "aliases"):
        values = entry.get(key, [])
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        aliases.update(str(value) for value in values if str(value).strip())

    cleaned = set()
    for name in aliases:
        name = str(name).strip()
        if not name:
            continue
        name = re.sub(r"\s*\(Leader\)", "", name, flags=re.I).strip()
        name = re.sub(r"\s*\(Banned\)", "", name, flags=re.I).strip()
        cleaned.add(name)

        match = re.match(r"^(.*?)\s*\(", name)
        if match:
            cleaned.add(match.group(1).strip())

    return {name for name in cleaned if name}


def _add_alias(alias_map, alias, standard_name):
    alias = str(alias or "").strip()
    standard_name = normalize_standard_name(standard_name)
    if alias and standard_name:
        alias_map.setdefault(alias, standard_name)


def _merge_entry(
    entries,
    standard_name,
    aliases=(),
    image="",
    attribute="",
    family="",
):
    standard_name = normalize_standard_name(standard_name)
    if not standard_name:
        return
    item = entries.setdefault(
        standard_name,
        {
            "aliases": set(),
            "image": "",
            "attribute": "",
            "family": "",
        },
    )
    item["aliases"].update(str(alias).strip() for alias in aliases if str(alias).strip())
    if image and not item["image"]:
        item["image"] = str(image).strip()
    if attribute and not item["attribute"]:
        item["attribute"] = str(attribute).strip()
    if family and not item["family"]:
        item["family"] = str(family).strip()


def generate_alias_and_images(
    catalog_path=CATALOG_PATH,
    luck_catalog_path=LUCK_CATALOG_PATH,
    swkit_alias_path=SWKIT_ALIAS_PATH,
    output_dir=JSON_DIR,
):
    catalog = _load_json(catalog_path, [])
    luck_catalog = _load_json(luck_catalog_path, [])
    swkit_alias_data = _load_json(swkit_alias_path, {})
    if not isinstance(catalog, list):
        catalog = []
    if not isinstance(luck_catalog, list):
        luck_catalog = []
    if not isinstance(swkit_alias_data, dict):
        swkit_alias_data = {}

    entries = {}
    for source_entry in catalog:
        standard_name = normalize_standard_name(source_entry.get("base_name"))
        if not standard_name:
            continue
        images = source_entry.get("images", [])
        image = images[0] if isinstance(images, list) and images else ""
        attributes = source_entry.get("attributes", [])
        families = source_entry.get("families", [])
        _merge_entry(
            entries,
            standard_name,
            aliases=make_aliases_for_entry(source_entry),
            image=image,
            attribute=_first_value(attributes),
            family=_first_value(families),
        )

    aliases_by_id = swkit_alias_data.get("aliases_by_id") or {}
    for source_entry in luck_catalog:
        display_name = normalize_standard_name(
            source_entry.get("slabel")
            or source_entry.get("label")
            or source_entry.get("slug")
        )
        if not display_name:
            continue
        unit_id = str(source_entry.get("com2us_id") or "").strip()
        aliases = list(make_aliases_for_entry(source_entry))
        aliases.extend(aliases_by_id.get(unit_id, []))
        image = _lucksack_image_url(source_entry.get("image"))
        attribute = _first_value(
            source_entry.get("element") or source_entry.get("attribute")
        )
        family = _first_value(
            source_entry.get("family")
            or source_entry.get("family_name")
            or source_entry.get("familyName")
        )
        _merge_entry(
            entries,
            display_name,
            aliases=aliases,
            image=image,
            attribute=attribute,
            family=family,
        )

    alias_map = {}
    image_map = {}
    attr_map = {}
    family_map = {}
    for standard_name, item in entries.items():
        for alias in item["aliases"] | {standard_name}:
            _add_alias(alias_map, alias, standard_name)
        if item["image"]:
            image_map[standard_name] = item["image"]
        if item["attribute"]:
            attr_map[standard_name] = item["attribute"]
        if item["family"]:
            family_map[standard_name] = item["family"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "monster_alias_map.json": alias_map,
        "monster_images.json": image_map,
        "monster_attributes.json": attr_map,
        "monster_families.json": family_map,
    }
    for filename, payload in outputs.items():
        with open(output_dir / filename, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Saved monster_alias_map.json ({len(alias_map)} aliases)")
    print(f"Saved monster_images.json ({len(image_map)} images)")
    print(f"Saved monster_attributes.json ({len(attr_map)} attributes)")
    print(f"Saved monster_families.json ({len(family_map)} families)")
    return outputs


if __name__ == "__main__":
    generate_alias_and_images()
