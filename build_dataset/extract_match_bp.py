import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_versions import extract_dataset_version_from_path, get_dataset_version_weight  # noqa: E402
from rank_tiers import enrich_rank_context, get_rank_tier_spec  # noqa: E402

try:
    import lxml  # noqa: F401
    HTML_PARSER = "lxml"
except Exception:
    HTML_PARSER = "html.parser"

GENERIC_PLAYER_LABELS = {
    "overview",
    "picks",
    "pick",
    "statistics",
    "stats",
    "builds",
    "runes",
    "artifacts",
    "battle history",
    "monster box",
    "match history",
    "leaderboard",
}
MIN_VALID_RATING = 800
MAX_VALID_RATING = 4000


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_name_from_alt(alt: str) -> str:
    alt = clean_text(alt)
    m = re.match(r"^(.*?)\s*\(", alt)
    if m:
        return m.group(1).strip()
    return alt


def extract_slug(href: str) -> Optional[str]:
    if not href:
        return None
    m = re.search(r"/monster/([^/?#]+)", href)
    return m.group(1).strip() if m else None


def extract_player_id(href: str) -> Optional[int]:
    if not href:
        return None
    m = re.search(r"/player/(\d+)", href)
    return int(m.group(1)) if m else None


def extract_rank_and_player_id_from_filename(path: str):
    filename = os.path.basename(path)
    m = re.match(r"(?P<rank>\d+|norank)_(?P<player_id>\d+)\.html$", filename)
    if not m:
        return None, None

    rank_raw = m.group("rank")
    player_id_raw = m.group("player_id")

    rank = None if rank_raw == "norank" else int(rank_raw)
    player_id = int(player_id_raw)
    return rank, player_id


def parse_rating_text(text: str) -> Optional[int]:
    text = clean_text(text)
    if text.isdigit():
        value = int(text)
        if MIN_VALID_RATING <= value <= MAX_VALID_RATING:
            return value
    return None

def clean_player_name(text: str) -> Optional[str]:
    text = clean_text(text)
    if not text:
        return None

    lowered = text.lower()
    if lowered in GENERIC_PLAYER_LABELS:
        return None
    if re.fullmatch(r"(overview|picks|statistics|stats|builds|runes|artifacts)", lowered):
        return None
    if text.isdigit():
        return None
    return text

def extract_rating_candidates(text: str) -> List[int]:
    values = []
    seen = set()
    for raw in re.findall(r"\b(\d{3,4})\b", text or ""):
        value = int(raw)
        if not (MIN_VALID_RATING <= value <= MAX_VALID_RATING):
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def parse_monster_anchor(a_tag) -> Optional[Dict[str, Any]]:
    href = a_tag.get("href", "")
    slug = extract_slug(href)
    if not slug:
        return None

    img = a_tag.find("img")
    if not img:
        return None

    alt = clean_text(img.get("alt", ""))
    title = ""

    title_node = a_tag.find(attrs={"title": True})
    if title_node:
        title = clean_text(title_node.get("title", ""))

    order = None
    for span in a_tag.find_all("span"):
        txt = clean_text(span.get_text())
        if txt.isdigit():
            order = int(txt)
            break

    full_text = f"{alt} {title}".lower()
    is_banned = "(banned)" in full_text
    is_leader = "(leader)" in full_text

    is_first_pick = False
    for span in a_tag.find_all("span"):
        txt = clean_text(span.get_text()).lower()
        if txt == "1st":
            is_first_pick = True
            break

    return {
        "slug": slug,
        "name": parse_name_from_alt(alt),
        "alt": alt,
        "title": title,
        "order": order,
        "is_banned": is_banned,
        "is_leader": is_leader,
        "is_first_pick": is_first_pick,
    }


def parse_side_monsters(container) -> List[Dict[str, Any]]:
    monsters = []
    for a in container.find_all("a", href=True):
        href = a.get("href", "")
        if "/monster/" not in href:
            continue
        item = parse_monster_anchor(a)
        if item:
            monsters.append(item)

    monsters = [m for m in monsters if m.get("order") is not None]
    monsters.sort(key=lambda x: x["order"])
    return monsters


def parse_player_block(container) -> Dict[str, Any]:
    player_links = []
    for a in container.find_all("a", href=True):
        href = a.get("href", "")
        if "/player/" in href:
            player_links.append(a)

    player_id = None
    player_name = None

    for a in player_links:
        href = a.get("href", "")
        pid = extract_player_id(href)
        text = clean_player_name(a.get_text())
        if pid:
            player_id = pid
            if text:
                player_name = text
                break

    rating = None
    spans = container.find_all(["span", "div"])
    for tag in spans:
        rating = parse_rating_text(tag.get_text())
        if rating is not None:
            break

    return {
        "player_id": player_id,
        "name": player_name,
        "rating": rating,
    }


def split_mobile_match_blocks(soup: BeautifulSoup):
    all_divs = soup.find_all("div")
    blocks = []

    for div in all_divs:
        text = clean_text(div.get_text(" ", strip=True)).lower()
        has_result = ("loss" in text or "defeat" in text or "win" in text or "victory" in text)

        monster_links = div.find_all("a", href=True)
        monster_count = sum(1 for a in monster_links if "/monster/" in a.get("href", ""))
        player_count = sum(1 for a in monster_links if "/player/" in a.get("href", ""))

        if has_result and monster_count >= 10 and player_count >= 1:
            blocks.append(div)

    unique_blocks = []
    for blk in blocks:
        for other in blocks:
            if blk is other:
                continue
            if blk in other.find_all("div"):
                continue
        unique_blocks.append(blk)

    dedup = []
    seen = set()
    for blk in unique_blocks:
        key = clean_text(blk.get_text(" ", strip=True))
        if key and key not in seen:
            seen.add(key)
            dedup.append(blk)

    return dedup


def parse_result_and_time(block) -> Dict[str, str]:
    text = clean_text(block.get_text(" ", strip=True))

    result = None
    time_ago = None

    text_lower = text.lower()
    if "victory" in text_lower or re.search(r"\bwin\b", text_lower):
        result = "win"
    elif "defeat" in text_lower or re.search(r"\bloss\b", text_lower):
        result = "loss"

    m = re.search(r"(\d+\s+(seconds?|minutes?|hours?|days?)\s+ago)", text_lower)
    if m:
        time_ago = m.group(1)

    return {
        "result": result,
        "time_ago": time_ago,
    }


def parse_match_block(block, page_player_id=None, match_index=0) -> Optional[Dict[str, Any]]:
    info = parse_result_and_time(block)

    monster_anchors = [a for a in block.find_all("a", href=True) if "/monster/" in a.get("href", "")]
    if len(monster_anchors) < 10:
        return None

    parsed_monsters = []
    for a in monster_anchors:
        item = parse_monster_anchor(a)
        if item:
            parsed_monsters.append(item)

    if len(parsed_monsters) < 10:
        return None

    enemy_picks = parsed_monsters[:5]
    my_picks = parsed_monsters[5:10]

    player_anchors = [a for a in block.find_all("a", href=True) if "/player/" in a.get("href", "")]
    player_infos = []

    for a in player_anchors:
        href = a.get("href", "")
        pid = extract_player_id(href)
        text = clean_player_name(a.get_text())
        if pid or text:
            player_infos.append({"player_id": pid, "name": text})

    self_info = {"player_id": page_player_id, "name": None, "rating": None}
    opp_info = {"player_id": None, "name": None, "rating": None}

    seen = set()
    dedup_infos = []
    for item in player_infos:
        key = (item.get("player_id"), item.get("name"))
        if key in seen:
            continue
        seen.add(key)
        dedup_infos.append(item)

    if page_player_id is not None:
        self_candidates = [x for x in dedup_infos if x.get("player_id") == page_player_id]
        opp_candidates = [x for x in dedup_infos if x.get("player_id") != page_player_id]

        if self_candidates:
            self_info["player_id"] = self_candidates[0].get("player_id") or page_player_id
            self_info["name"] = self_candidates[0].get("name")
        if opp_candidates:
            opp_info["player_id"] = opp_candidates[0].get("player_id")
            opp_info["name"] = opp_candidates[0].get("name")

    if self_info["name"] is None or opp_info["name"] is None:
        named_infos = [x for x in dedup_infos if x.get("name")]
        if len(named_infos) >= 2:
            if opp_info["name"] is None:
                opp_info["player_id"] = named_infos[0].get("player_id")
                opp_info["name"] = named_infos[0].get("name")
            if self_info["name"] is None:
                self_info["player_id"] = named_infos[1].get("player_id") or page_player_id
                self_info["name"] = named_infos[1].get("name")
        elif len(named_infos) == 1:
            only_info = named_infos[0]
            if only_info.get("player_id") == page_player_id:
                self_info["player_id"] = only_info.get("player_id") or page_player_id
                self_info["name"] = only_info.get("name")
            else:
                opp_info["player_id"] = only_info.get("player_id")
                opp_info["name"] = only_info.get("name")

    ratings = extract_rating_candidates(block.get_text(" ", strip=True))
    if len(ratings) >= 2:
        opp_info["rating"] = ratings[0]
        self_info["rating"] = ratings[1]
    elif len(ratings) == 1:
        if self_info["name"] and not opp_info["name"]:
            self_info["rating"] = ratings[0]
        elif opp_info["name"] and not self_info["name"]:
            opp_info["rating"] = ratings[0]

    return {
        "page_player_id": page_player_id,
        "match_index": match_index,
        "result": info["result"],
        "time_ago": info["time_ago"],
        "opponent": opp_info,
        "self": self_info,
        "enemy_picks": enemy_picks,
        "my_picks": my_picks,
    }


def normalize_match_for_dataset(match: Dict[str, Any], version_weights=None) -> Dict[str, Any]:
    enemy_sorted = sorted(match["enemy_picks"], key=lambda x: x["order"] or 999)
    my_sorted = sorted(match["my_picks"], key=lambda x: x["order"] or 999)

    my_ban = next((m["name"] for m in my_sorted if m["is_banned"]), None)
    enemy_ban = next((m["name"] for m in enemy_sorted if m["is_banned"]), None)
    rank_context = enrich_rank_context(
        rank=match.get("player_rank"),
        tier_key=match.get("rank_tier"),
        explicit_weight=match.get("rank_weight"),
    )
    dataset_version = match.get("dataset_version")
    version_weight = get_dataset_version_weight(
        version=dataset_version,
        version_weights=version_weights,
        explicit_weight=match.get("version_weight"),
    )
    sample_weight = rank_context["rank_weight"] * version_weight

    return {
        "player_id": match.get("page_player_id"),
        "player_rank": rank_context["player_rank"],
        "rank_tier": rank_context["rank_tier"],
        "rank_tier_label": rank_context["rank_tier_label"],
        "rank_weight": rank_context["rank_weight"],
        "dataset_version": dataset_version,
        "version_weight": version_weight,
        "sample_weight": sample_weight,
        "match_index": match.get("match_index"),
        "result": match.get("result"),
        "time_ago": match.get("time_ago"),
        "self_name": match.get("self", {}).get("name"),
        "self_rating": match.get("self", {}).get("rating"),
        "opponent_name": match.get("opponent", {}).get("name"),
        "opponent_rating": match.get("opponent", {}).get("rating"),
        "my_pick_order": [m["name"] for m in my_sorted],
        "enemy_pick_order": [m["name"] for m in enemy_sorted],
        "my_pick_slugs": [m["slug"] for m in my_sorted],
        "enemy_pick_slugs": [m["slug"] for m in enemy_sorted],
        "my_ban": my_ban,
        "enemy_ban": enemy_ban,
        "my_leader": next((m["name"] for m in my_sorted if m["is_leader"]), None),
        "enemy_leader": next((m["name"] for m in enemy_sorted if m["is_leader"]), None),
        "my_first_pick": next((m["name"] for m in my_sorted if m["is_first_pick"] or m["order"] == 1), None),
        "enemy_first_pick": next((m["name"] for m in enemy_sorted if m["is_first_pick"] or m["order"] == 1), None),
        "source_html": match.get("source_html"),
    }


def extract_page_player_id_from_filename(path: str) -> Optional[int]:
    _, player_id = extract_rank_and_player_id_from_filename(path)
    return player_id


def extract_page_rank_from_filename(path: str) -> Optional[int]:
    rank, _ = extract_rank_and_player_id_from_filename(path)
    return rank


def extract_rank_tier_from_path(path: str) -> Optional[str]:
    parent_name = Path(path).resolve().parent.name
    spec = get_rank_tier_spec(tier_key=parent_name)
    if spec["key"] == parent_name:
        return spec["key"]
    return None


def process_html_file(html_path: str, dataset_root=None, version_weights=None):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, HTML_PARSER)
    page_player_id = extract_page_player_id_from_filename(html_path)
    page_rank = extract_page_rank_from_filename(html_path)
    path_rank_tier = extract_rank_tier_from_path(html_path)
    rank_context = enrich_rank_context(rank=page_rank, tier_key=path_rank_tier)
    dataset_version = extract_dataset_version_from_path(
        html_path,
        dataset_root=dataset_root or (ROOT_DIR / "dataset"),
    )
    version_weight = get_dataset_version_weight(dataset_version, version_weights=version_weights)
    blocks = split_mobile_match_blocks(soup)

    raw_matches = []
    dataset_matches = []

    for idx, block in enumerate(blocks, start=1):
        match = parse_match_block(block, page_player_id=page_player_id, match_index=idx)
        if not match:
            continue

        if len(match["my_picks"]) == 5 and len(match["enemy_picks"]) == 5:
            match["player_rank"] = rank_context["player_rank"]
            match["rank_tier"] = rank_context["rank_tier"]
            match["rank_tier_label"] = rank_context["rank_tier_label"]
            match["rank_weight"] = rank_context["rank_weight"]
            match["dataset_version"] = dataset_version
            match["version_weight"] = version_weight
            match["sample_weight"] = rank_context["rank_weight"] * version_weight
            match["source_html"] = os.path.relpath(html_path, start=os.getcwd())
            raw_matches.append(match)
            dataset_matches.append(normalize_match_for_dataset(match, version_weights=version_weights))

    return raw_matches, dataset_matches


if __name__ == "__main__":
    html_path = "D:\\jupyter\\auto_bp\\dataset\\1.html"

    raw_matches, dataset_matches = process_html_file(html_path)

    output_dir = ROOT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(raw_matches, output_dir / "extract_match_raw_smoke.pt")
    torch.save(dataset_matches, output_dir / "extract_match_dataset_smoke.pt")

    print(f"raw matches: {len(raw_matches)}")
    print(f"dataset matches: {len(dataset_matches)}")
    print("saved: data/extract_match_raw_smoke.pt, data/extract_match_dataset_smoke.pt")
