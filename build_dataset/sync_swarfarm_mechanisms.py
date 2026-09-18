from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_ROOT = "https://swarfarm.com/api/v2"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "json" / "lucksack_monster_catalog.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "json" / "sw_mechanism_profiles.json"
OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
)

EFFECT_TAGS = {
    "stun": {"control"},
    "freeze": {"control"},
    "sleep": {"control"},
    "provoke": {"control"},
    "silence": {"control"},
    "seal": {"control"},
    "suppression": {"control"},
    "decrease atb": {"atb_control"},
    "absorb atb": {"atb_control"},
    "decrease atb effects": {"atb_control"},
    "increase atb": {"tempo"},
    "increases atb effects": {"tempo"},
    "additional turn": {"tempo"},
    "increase atk spd": {"speed"},
    "decrease atk spd": {"speed_control"},
    "increase cooltime": {"cooldown_control"},
    "reduce cooltime": {"cooldown_reduce"},
    "remove buff": {"strip"},
    "steal buff": {"strip"},
    "beneficial effects blocked": {"buff_block"},
    "decrease buff duration": {"buff_control"},
    "increase buff duration": {"buff_support"},
    "cleanse": {"cleanse"},
    "remove inability effect": {"cleanse"},
    "immunity": {"immunity"},
    "immune to harmful effects": {"control_immunity", "immunity"},
    "immune to inability effects": {"control_immunity"},
    "immune to stun": {"control_immunity"},
    "immune to freeze": {"control_immunity"},
    "immune to sleep": {"control_immunity"},
    "immune to atb decreasing effects": {"atb_immunity"},
    "blocked attack bar increasing effects": {"atb_immunity"},
    "immune against cool down effects": {"cooldown_immunity"},
    "immune to continuous dmg": {"dot_immunity"},
    "bomb": {"bomb_dot"},
    "detonate bomb": {"bomb_dot"},
    "continuous dmg": {"bomb_dot"},
    "detonate continuous damage": {"bomb_dot"},
    "heal": {"heal"},
    "recovery": {"heal"},
    "self-heal": {"heal"},
    "increase heal amount": {"heal"},
    "redistribute hp": {"heal"},
    "revive": {"revive"},
    "soul stone": {"revive"},
    "soul protection": {"damage_reduction"},
    "unrecoverable": {"anti_heal"},
    "anti-revive": {"anti_revive"},
    "ignores resist death": {"anti_revive"},
    "unrevivable": {"anti_revive"},
    "curse of death": {"anti_revive"},
    "destroy hp": {"hp_destroy"},
    "hp reduction": {"hp_destroy"},
    "decrease damage": {"damage_reduction"},
    "shield": {"damage_reduction"},
    "invincible": {"damage_reduction"},
    "endure": {"damage_reduction"},
    "defend": {"damage_reduction"},
    "ignore damage reduction": {"ignore_damage_reduction"},
    "ignore def": {"ignore_defense"},
    "ignore resistance": {"ignore_resistance"},
    "irresistible": {"ignore_resistance"},
    "decrease def": {"def_break"},
    "increase damage": {"damage_amp"},
    "brand": {"damage_amp"},
    "debuff bonus damage": {"damage_amp"},
    "counter": {"counter"},
    "reflect dmg": {"counter"},
    "passive reflect dmg": {"counter"},
    "harmful effect removal blocked": {"cleanse_block"},
    "prevent harmful effect reapply": {"cleanse_block"},
}

TEXT_TAG_RULES = (
    (r"\battack bar\b.*\b(absorb|decrease|reduce)\b", "atb_control"),
    (r"\b(absorb|decrease|reduce)\b.*\battack bar\b", "atb_control"),
    (r"\b(increase|fill)\b.*\battack bar\b", "tempo"),
    (r"\b(additional turn|gain another turn)\b", "tempo"),
    (r"\b(remove|removes|removing)\b.*\bharmful effect", "cleanse"),
    (r"\b(immunity|immune against|immune to)\b", "immunity"),
    (r"\b(revive|revives|soul stone)\b", "revive"),
    (r"\b(continuous damage|damage over time|bomb)\b", "bomb_dot"),
    (r"\b(cooldown|cool time)\b", "cooldown_control"),
    (r"\b(heal|heals|recover|recovery)\b", "heal"),
    (r"\b(remove|steal)\b.*\bbeneficial effect", "strip"),
)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "autobp/1.0"})
    with OPENER.open(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all(endpoint):
    url = f"{API_ROOT}/{endpoint}/?page_size=100"
    rows = []
    page_count = 0
    with tqdm(
        desc=f"SWARFARM {endpoint}",
        unit="row",
        dynamic_ncols=True,
    ) as progress:
        while url:
            payload = fetch_json(url)
            page_count += 1
            page_rows = list(payload.get("results") or [])
            rows.extend(page_rows)
            if progress.total is None and payload.get("count") is not None:
                progress.total = int(payload["count"])
                progress.refresh()
            progress.update(len(page_rows))
            progress.set_postfix_str(f"pages={page_count}")
            url = payload.get("next")
    return rows


def _effect_name(effect):
    nested = effect.get("effect") if isinstance(effect, dict) else None
    if isinstance(nested, dict):
        return str(nested.get("name") or "").strip()
    return str(nested or "").strip()


def _tags_for_effect(name):
    normalized = str(name or "").strip().lower()
    return set(EFFECT_TAGS.get(normalized, set()))


def _tags_for_text(text):
    tags = set()
    value = str(text or "").lower()
    for pattern, tag in TEXT_TAG_RULES:
        if re.search(pattern, value):
            tags.add(tag)
    return tags


def build_profile(monster, skills_by_id):
    tags = set()
    skill_rows = []
    all_text = []
    effect_names = set()
    for skill_id in monster.get("skills") or []:
        skill = skills_by_id.get(int(skill_id))
        if not skill:
            continue
        description = str(skill.get("description") or "")
        name = str(skill.get("name") or "")
        all_text.extend((name, description))
        skill_effects = []
        for effect in skill.get("effects") or []:
            effect_name = _effect_name(effect)
            if not effect_name:
                continue
            effect_names.add(effect_name)
            skill_effects.append(effect_name)
            tags.update(_tags_for_effect(effect_name))
        tags.update(_tags_for_text(name))
        tags.update(_tags_for_text(description))
        if bool(skill.get("passive")):
            tags.add("passive")
        if bool(skill.get("aoe")) or any(bool(effect.get("all")) for effect in skill.get("effects") or []):
            tags.add("aoe")
        if int(skill.get("hits") or 0) > 1:
            tags.add("multi_hit")
        skill_rows.append(
            {
                "name": name,
                "slot": skill.get("slot"),
                "passive": bool(skill.get("passive")),
                "aoe": bool(skill.get("aoe")),
                "hits": int(skill.get("hits") or 0),
                "cooltime": int(skill.get("cooltime") or 0),
                "max_level": int(skill.get("max_level") or 0),
                "effects": sorted(set(skill_effects)),
                "scales_with": list(skill.get("scales_with") or []),
                "multiplier_formula": str(skill.get("multiplier_formula") or ""),
            }
        )

    leader_skill = monster.get("leader_skill") or {}
    leader_attribute = str(leader_skill.get("attribute") or "").strip().lower()
    if "speed" in leader_attribute:
        tags.add("speed")
        tags.add("leader_speed")
    if "attack bar" in leader_attribute:
        tags.add("tempo")

    threat_tags = set()
    if "control" in tags:
        threat_tags.add("control")
    if "atb_control" in tags:
        threat_tags.add("atb_control")
    if "cooldown_control" in tags:
        threat_tags.add("cooldown_control")
    if tags & {"heal", "revive"}:
        threat_tags.add("sustain")
    if "revive" in tags:
        threat_tags.add("revive")
    if "bomb_dot" in tags:
        threat_tags.add("bomb_dot")
    if tags & {"damage_reduction", "immunity", "shield"}:
        threat_tags.add("defense")
    if tags & {"tempo", "speed"}:
        threat_tags.add("tempo")

    return {
        "name": str(monster.get("name") or ""),
        "com2us_id": str(monster.get("com2us_id") or ""),
        "element": str(monster.get("element") or ""),
        "archetype": str(monster.get("archetype") or ""),
        "base_stats": {
            "hp": int(monster.get("base_hp") or 0),
            "attack": int(monster.get("base_attack") or 0),
            "defense": int(monster.get("base_defense") or 0),
            "speed": int(monster.get("speed") or 0),
            "crit_rate": int(monster.get("crit_rate") or 0),
            "crit_damage": int(monster.get("crit_damage") or 0),
            "resistance": int(monster.get("resistance") or 0),
            "accuracy": int(monster.get("accuracy") or 0),
        },
        "tags": sorted(tags),
        "threat_tags": sorted(threat_tags),
        "effects": sorted(effect_names),
        "skills": skill_rows,
        "leader_skill": leader_skill,
        "source_monster_id": monster.get("id"),
    }


def build_snapshot(monsters, skills, catalog):
    skills_by_id = {int(skill["id"]): skill for skill in skills if skill.get("id") is not None}
    monsters_by_com2us_id = {}
    for monster in monsters:
        unit_id = str(monster.get("com2us_id") or "")
        if unit_id:
            monsters_by_com2us_id.setdefault(unit_id, monster)

    profiles_by_slug = {}
    aliases_by_slug = {}
    matched_slugs = set()
    for item in tqdm(catalog, desc="构建机制画像", unit="monster", dynamic_ncols=True):
        slug = str(item.get("slug") or "").strip().lower()
        unit_id = str(item.get("com2us_id") or "")
        if not slug or not unit_id:
            continue
        monster = monsters_by_com2us_id.get(unit_id)
        profile = build_profile(monster or {"com2us_id": unit_id, "name": item.get("slabel")}, skills_by_id)
        profile["slug"] = slug
        profile["name"] = str(item.get("slabel") or profile.get("name") or slug)
        profile["element"] = str(item.get("element") or profile.get("element") or "")
        profiles_by_slug[slug] = profile
        for alias in (slug, item.get("slabel"), item.get("label"), profile.get("name")):
            if alias:
                aliases_by_slug[str(alias).strip().lower()] = slug
        if monster:
            matched_slugs.add(slug)

    return {
        "schema_version": 2,
        "source": "swarfarm.com",
        "source_url": f"{API_ROOT}/",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "monster_count": len(monsters),
        "skill_count": len(skills),
        "profile_count": len(profiles_by_slug),
        "matched_count": len(matched_slugs),
        "aliases_by_slug": aliases_by_slug,
        "profiles_by_slug": profiles_by_slug,
    }


def sync_swarfarm_mechanisms(
    catalog_path=DEFAULT_CATALOG_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
):
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    monsters = fetch_all("monsters")
    skills = fetch_all("skills")
    snapshot = build_snapshot(monsters, skills, catalog)
    if not snapshot["profiles_by_slug"]:
        raise RuntimeError("SWARFARM mechanism snapshot is empty.")
    save_json(snapshot, output_path)
    print(
        f"[OK] saved mechanisms: {output_path} "
        f"({snapshot['matched_count']}/{snapshot['profile_count']} matched)"
    )
    return snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Sync SWARFARM monster skill mechanisms.")
    parser.add_argument("--catalog-path", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    sync_swarfarm_mechanisms(
        catalog_path=args.catalog_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
