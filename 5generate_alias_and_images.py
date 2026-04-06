import json
import re

def normalize_standard_name(name: str) -> str:
    return name.strip()

def make_aliases_for_entry(entry):
    aliases = set()
    aliases.add(entry["base_name"])

    for x in entry.get("raw_names", []):
        aliases.add(x)
    for x in entry.get("alts", []):
        aliases.add(x)
    for x in entry.get("titles", []):
        aliases.add(x)

    cleaned = set()
    for name in aliases:
        n = name.strip()
        if not n:
            continue

        n = re.sub(r"\s*\(Leader\)", "", n, flags=re.I).strip()
        n = re.sub(r"\s*\(Banned\)", "", n, flags=re.I).strip()

        cleaned.add(n)

        m = re.match(r"^(.*?)\s*\(", n)
        if m:
            cleaned.add(m.group(1).strip())

    return cleaned

def main():
    with open("json//monster_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    alias_map = {}
    image_map = {}
    attr_map = {}
    family_map = {}

    for entry in catalog:
        standard_name = normalize_standard_name(entry["base_name"])
        aliases = make_aliases_for_entry(entry)

        for alias in aliases:
            alias_map[alias] = standard_name

        images = entry.get("images", [])
        if images:
            image_map[standard_name] = images[0]

        attrs = entry.get("attributes", [])
        if attrs:
            attr_map[standard_name] = attrs[0]

        families = entry.get("families", [])
        if families:
            family_map[standard_name] = families[0]

    with open("json//monster_alias_map.json", "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)

    with open("json//monster_images.json", "w", encoding="utf-8") as f:
        json.dump(image_map, f, ensure_ascii=False, indent=2)

    with open("json//monster_attributes.json", "w", encoding="utf-8") as f:
        json.dump(attr_map, f, ensure_ascii=False, indent=2)

    with open("json//monster_families.json", "w", encoding="utf-8") as f:
        json.dump(family_map, f, ensure_ascii=False, indent=2)

    print("Saved monster_alias_map.json")
    print("Saved monster_images.json")
    print("Saved monster_attributes.json")
    print("Saved monster_families.json")

if __name__ == "__main__":
    main()