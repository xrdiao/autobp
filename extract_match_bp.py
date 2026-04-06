import re
import json
import os
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def parse_name_from_alt(alt: str) -> str:
    alt = clean_text(alt)
    # 优先取括号前
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

def parse_rating_text(text: str) -> Optional[int]:
    text = clean_text(text)
    if text.isdigit():
        return int(text)
    return None

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

    # title 通常在内部 div 上
    title_node = a_tag.find(attrs={"title": True})
    if title_node:
        title = clean_text(title_node.get("title", ""))

    # order 数字
    order = None
    for span in a_tag.find_all("span"):
        txt = clean_text(span.get_text())
        if txt.isdigit():
            order = int(txt)
            break

    full_text = f"{alt} {title}".lower()

    is_banned = "(banned)" in full_text
    is_leader = "(leader)" in full_text

    # 是否有 first-pick 标签
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
        "is_first_pick": is_first_pick
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

    # 按 order 排序
    monsters = [m for m in monsters if m.get("order") is not None]
    monsters.sort(key=lambda x: x["order"])
    return monsters

def parse_player_block(container) -> Dict[str, Any]:
    """
    解析一侧玩家信息：
    - player_id
    - player_name
    - rating
    """
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
        text = clean_text(a.get_text())
        if pid:
            player_id = pid
            if text:
                player_name = text
                break

    # 取可能的 rating
    rating = None
    spans = container.find_all(["span", "div"])
    for tag in spans:
        txt = clean_text(tag.get_text())
        if txt.isdigit() and len(txt) in (3, 4):
            rating = int(txt)
            break

    return {
        "player_id": player_id,
        "name": player_name,
        "rating": rating
    }

def split_mobile_match_blocks(soup: BeautifulSoup):
    """
    优先针对你给的这类移动端结构：
    lg:hidden 下，每个比赛通常是一个较大的外层 div。
    这里我们通过包含：
    - result文本
    - 两侧 monster 链接
    来判断一个比赛块
    """
    all_divs = soup.find_all("div")
    blocks = []

    for div in all_divs:
        text = clean_text(div.get_text(" ", strip=True)).lower()

        # 结果关键词
        has_result = ("loss" in text or "defeat" in text or "win" in text or "victory" in text)

        monster_links = div.find_all("a", href=True)
        monster_count = sum(1 for a in monster_links if "/monster/" in a.get("href", ""))

        player_count = sum(1 for a in monster_links if "/player/" in a.get("href", ""))

        # 一个比赛块通常至少有 10 个 monster + 2 个玩家
        if has_result and monster_count >= 10 and player_count >= 1:
            blocks.append(div)

    # 去掉明显嵌套重复：保留较小的“合理块”
    unique_blocks = []
    for blk in blocks:
        is_nested = False
        for other in blocks:
            if blk is other:
                continue
            if blk in other.find_all("div"):
                # 如果 blk 是 other 的子 div，保留更小的 blk
                continue
        unique_blocks.append(blk)

    # 进一步去重：按文本长度
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

    # result
    text_lower = text.lower()
    if "victory" in text_lower or re.search(r"\bwin\b", text_lower):
        result = "win"
    elif "defeat" in text_lower or re.search(r"\bloss\b", text_lower):
        result = "loss"

    # time
    m = re.search(r"(\d+\s+(seconds?|minutes?|hours?|days?)\s+ago)", text_lower)
    if m:
        time_ago = m.group(1)

    return {
        "result": result,
        "time_ago": time_ago
    }

def parse_match_block(block, page_player_id=None, match_index=0) -> Optional[Dict[str, Any]]:
    info = parse_result_and_time(block)

    # 找所有 monster anchor
    monster_anchors = [a for a in block.find_all("a", href=True) if "/monster/" in a.get("href", "")]
    if len(monster_anchors) < 10:
        return None

    # 移动端结构一般是：
    # 对手信息 + 对手5个monster + 中间 swords + 我方信息 + 我方5个monster
    # 所以简单做法：按出现顺序切前5和后5
    parsed_monsters = []
    for a in monster_anchors:
        item = parse_monster_anchor(a)
        if item:
            parsed_monsters.append(item)

    if len(parsed_monsters) < 10:
        return None

    enemy_picks = parsed_monsters[:5]
    my_picks = parsed_monsters[5:10]

    # 尝试提取玩家信息
    player_anchors = [a for a in block.find_all("a", href=True) if "/player/" in a.get("href", "")]
    player_infos = []

    for a in player_anchors:
        href = a.get("href", "")
        pid = extract_player_id(href)
        text = clean_text(a.get_text())
        if pid or text:
            player_infos.append((pid, text))

    # 简单找 page_player_id 对应我方
    self_info = {"player_id": page_player_id, "name": None, "rating": None}
    opp_info = {"player_id": None, "name": None, "rating": None}

    # 尝试从文本中取两个名字
    names = []
    for pid, text in player_infos:
        if text:
            names.append((pid, text))

    # 去重
    seen = set()
    dedup_names = []
    for x in names:
        if x not in seen:
            seen.add(x)
            dedup_names.append(x)

    if len(dedup_names) >= 2:
        # 一般前一个是对手，后一个是自己
        opp_info["player_id"] = dedup_names[0][0]
        opp_info["name"] = dedup_names[0][1]
        self_info["player_id"] = dedup_names[1][0] or page_player_id
        self_info["name"] = dedup_names[1][1]
    elif len(dedup_names) == 1:
        # 如果只有一个能读出来，默认是对手
        opp_info["player_id"] = dedup_names[0][0]
        opp_info["name"] = dedup_names[0][1]

    # ratings：提取块内所有3~4位数字，前两个大概率是双方rating
    numbers = re.findall(r"\b(\d{3,4})\b", block.get_text(" ", strip=True))
    numbers = [int(x) for x in numbers]

    if len(numbers) >= 2:
        opp_info["rating"] = numbers[0]
        self_info["rating"] = numbers[1]

    return {
        "page_player_id": page_player_id,
        "match_index": match_index,
        "result": info["result"],
        "time_ago": info["time_ago"],
        "opponent": opp_info,
        "self": self_info,
        "enemy_picks": enemy_picks,
        "my_picks": my_picks
    }

def normalize_match_for_dataset(match: Dict[str, Any]) -> Dict[str, Any]:
    enemy_sorted = sorted(match["enemy_picks"], key=lambda x: x["order"] or 999)
    my_sorted = sorted(match["my_picks"], key=lambda x: x["order"] or 999)

    my_ban = next((m["name"] for m in my_sorted if m["is_banned"]), None)
    enemy_ban = next((m["name"] for m in enemy_sorted if m["is_banned"]), None)

    return {
        "player_id": match.get("page_player_id"),
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
    }

def extract_page_player_id_from_filename(path: str) -> Optional[int]:
    filename = os.path.basename(path)
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None

def process_html_file(html_path: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    page_player_id = extract_page_player_id_from_filename(html_path)

    blocks = split_mobile_match_blocks(soup)

    raw_matches = []
    dataset_matches = []

    for idx, block in enumerate(blocks, start=1):
        match = parse_match_block(block, page_player_id=page_player_id, match_index=idx)
        if not match:
            continue

        # 只保留刚好 5v5 的
        if len(match["my_picks"]) == 5 and len(match["enemy_picks"]) == 5:
            raw_matches.append(match)
            dataset_matches.append(normalize_match_for_dataset(match))

    return raw_matches, dataset_matches

if __name__ == "__main__":
    html_path = "D:\\jupyter\\auto_bp\\dataset\\1.html"   # 改成你的文件名

    raw_matches, dataset_matches = process_html_file(html_path)

    with open("matches_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_matches, f, ensure_ascii=False, indent=2)

    with open("bp_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_matches, f, ensure_ascii=False, indent=2)

    print(f"raw matches: {len(raw_matches)}")
    print(f"dataset matches: {len(dataset_matches)}")
    print("saved: matches_raw.json, bp_dataset.json")