from __future__ import annotations

import argparse
import asyncio
import ssl
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COLLECTION_URL = "https://www.swkit.fun/tools/collection"
TIER_LIST_URL = "https://www.swkit.fun/tools/tier-list"
SWKIT_BASE_URL = "https://www.swkit.fun/"
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "json" / "swkit_monster_aliases.json"
DEFAULT_TIER_PATH = PROJECT_ROOT / "json" / "swkit_rta_tiers.json"
NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def fetch_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with NO_PROXY_OPENER.open(request, timeout=60) as response:
        return response.read().decode("utf-8")


def fetch_json(url):
    return json.loads(fetch_text(url))


def first_match(pattern, text, label):
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"could not find {label}")
    return match.group(1)


def extract_array_literal(script_text, marker_pattern):
    marker = re.search(marker_pattern, script_text)
    if not marker:
        return "[]"
    start = script_text.find("[", marker.start())
    depth = 0
    quote = ""
    escape = False
    for index in range(start, len(script_text)):
        char = script_text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return script_text[start : index + 1]
    return "[]"


def parse_tier_entries(script_text):
    array_literal = extract_array_literal(script_text, r"=\[\{id:\d+,tier:")
    json_text = array_literal.replace("`", '"')
    json_text = re.sub(r"([{\[,])([A-Za-z_][A-Za-z0-9_]*):", r'\1"\2":', json_text)
    json_text = re.sub(r"([:,\[])\.(\d+)", r"\g<1>0.\2", json_text)
    json_text = json_text.replace("!0", "true").replace("!1", "false")
    return json.loads(json_text)


def parse_tier_date(script_text):
    match = re.search(r"var\s+[A-Za-z_$][\w$]*=`([^`]+)`", script_text)
    return match.group(1) if match else ""


def add_alias(aliases_by_id, unit_id, alias):
    alias = str(alias or "").strip()
    if not alias:
        return
    aliases_by_id.setdefault(str(unit_id), set()).add(alias)


def build_alias_snapshot(monsters, custom_names, collab_pairs, source_urls):
    aliases_by_id = {}
    monster_by_id = {}
    for monster in monsters:
        unit_id = str(monster.get("unit_master_id") or "")
        if not unit_id:
            continue
        monster_by_id[unit_id] = monster
        add_alias(aliases_by_id, unit_id, monster.get("name_en"))
        add_alias(aliases_by_id, unit_id, custom_names.get(unit_id))

    for pair in collab_pairs:
        left_id = str(pair.get("id_a") or "")
        right_id = str(pair.get("id_b") or "")
        for source_id, target_id, name_key in (
            (left_id, right_id, "name_a"),
            (right_id, left_id, "name_b"),
        ):
            if not source_id or not target_id:
                continue
            add_alias(aliases_by_id, source_id, pair.get(name_key))
            add_alias(aliases_by_id, source_id, custom_names.get(target_id))
            add_alias(aliases_by_id, source_id, (monster_by_id.get(target_id) or {}).get("name_en"))

    cleaned = {
        unit_id: sorted(aliases)
        for unit_id, aliases in sorted(aliases_by_id.items(), key=lambda item: int(item[0]))
        if aliases
    }
    return {
        "schema_version": 1,
        "source": "swkit.fun",
        "source_urls": source_urls,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "aliases_by_id": cleaned,
        "alias_count": sum(len(value) for value in cleaned.values()),
    }


def build_tier_snapshot(tier_entries, tier_date, monsters, custom_names, tier_script_url):
    monster_by_id = {str(monster.get("unit_master_id")): monster for monster in monsters}
    entries = []
    for entry in tier_entries:
        unit_id = str(entry.get("id") or "")
        monster = monster_by_id.get(unit_id) or {}
        enriched = dict(entry)
        enriched["id"] = unit_id
        enriched["name_en"] = monster.get("name_en", "")
        enriched["name_cn"] = custom_names.get(unit_id, "")
        enriched["element"] = monster.get("element", "")
        enriched["natural_stars"] = monster.get("natural_stars")
        entries.append(enriched)
    return {
        "schema_version": 1,
        "source": "swkit.fun",
        "source_url": TIER_LIST_URL,
        "tier_script_url": tier_script_url,
        "tier_date": tier_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "entries_by_id": {entry["id"]: entry for entry in entries},
    }


def collect_swkit_data():
    html = fetch_text(COLLECTION_URL)
    main_asset = first_match(r'src="(/assets/main-[^"]+\.js)"', html, "main asset")
    use_store_asset = first_match(r'href="(/assets/useSwStore-[^"]+\.js)"', html, "store asset")
    account_asset = first_match(r'href="(/assets/accountAnalysis-[^"]+\.js)"', html, "account analysis asset")
    main_script = fetch_text(urljoin(SWKIT_BASE_URL, main_asset))
    store_script = fetch_text(urljoin(SWKIT_BASE_URL, use_store_asset))
    account_script = fetch_text(urljoin(SWKIT_BASE_URL, account_asset))
    tier_asset = first_match(r"(TierList-[A-Za-z0-9_-]+\.js)", main_script, "tier asset")
    monster_file = first_match(r"(sw_monsters_dict-[A-Za-z0-9_-]+\.json)", store_script, "monster dict")
    custom_file = first_match(r"(sw_custom_names\.json)", store_script, "custom names")
    collab_file = first_match(r"(sw_collab_pairs-[A-Za-z0-9_-]+\.json)", account_script, "collab pairs")

    monster_url = f"https://oss.swkit.fun/data/{monster_file}"
    custom_url = f"https://oss.swkit.fun/data/{custom_file}"
    collab_url = f"https://oss.swkit.fun/data/{collab_file}"
    tier_script_url = urljoin(SWKIT_BASE_URL, f"/assets/{tier_asset}")

    monsters = fetch_json(monster_url)
    custom_names = fetch_json(custom_url)
    collab_pairs = fetch_json(collab_url)
    tier_script = fetch_text(tier_script_url)

    aliases = build_alias_snapshot(
        monsters,
        custom_names,
        collab_pairs,
        {
            "collection": COLLECTION_URL,
            "monsters": monster_url,
            "custom_names": custom_url,
            "collab_pairs": collab_url,
        },
    )
    tiers = build_tier_snapshot(
        parse_tier_entries(tier_script),
        parse_tier_date(tier_script),
        monsters,
        custom_names,
        tier_script_url,
    )
    return aliases, tiers


async def sync_swkit_data(alias_path=DEFAULT_ALIAS_PATH, tier_path=DEFAULT_TIER_PATH):
    aliases, tiers = collect_swkit_data()
    if not tiers["entries"]:
        raise RuntimeError("SWKit RTA tier list is empty; refusing to save an empty snapshot.")
    save_json(aliases, alias_path)
    save_json(tiers, tier_path)
    print(f"[OK] saved aliases: {alias_path} ({aliases['alias_count']} aliases)")
    print(f"[OK] saved RTA tiers: {tier_path} ({len(tiers['entries'])} entries, {tiers['tier_date']})")
    return aliases, tiers


def parse_args():
    parser = argparse.ArgumentParser(description="Sync SWKit monster aliases and RTA tier data.")
    parser.add_argument("--alias-path", default=str(DEFAULT_ALIAS_PATH))
    parser.add_argument("--tier-path", default=str(DEFAULT_TIER_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(sync_swkit_data(alias_path=args.alias_path, tier_path=args.tier_path))


if __name__ == "__main__":
    main()
