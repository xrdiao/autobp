import os
import re
import json
from bs4 import BeautifulSoup
from pathlib import Path

DATASET_DIR = "dataset"
IMAGES_DIR = "images"
OUTPUT_PATH = "json//monster_images.json"
LUCK_CATALOG_PATH = "json//lucksack_monster_catalog.json"
MISSING_REPORT_PATH = "json//missing_images_report.json"

# 只接受 unit_icon 开头的图片
UNIT_ICON_PATTERN = re.compile(r"^unit_icon.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

try:
    __import__("lxml")
    HTML_PARSER = "lxml"
except Exception:
    HTML_PARSER = "html.parser"


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def extract_base_name(text: str):
    """
    从 alt/title 提取基础魔灵名
    例如:
    - Oliver (Wind Sky Surfer) -> Oliver
    - Josephine (Water Paladin) (Leader) -> Josephine
    - Ereshion (Wind Drakan Warrior) (Banned) -> Ereshion
    """
    text = clean_text(text)

    # 去掉常见状态后缀
    text = re.sub(r"\s*\(Leader\)", "", text, flags=re.I)
    text = re.sub(r"\s*\(Banned\)", "", text, flags=re.I)

    # 如果是 xxx (Family) 形式，取前半段
    m = re.match(r"^(.*?)\s*\(", text)
    if m:
        return m.group(1).strip()

    return text.strip()

def is_unit_icon(src: str):
    """
    只保留 unit_icon 开头的目标头像
    """
    if not src:
        return False
    filename = os.path.basename(src.replace("\\", "/"))
    return bool(UNIT_ICON_PATTERN.match(filename))

def image_src_to_images_path(src: str):
    """
    将 html 中的 src 转换为 images/<filename>
    例如:
    ./240415_files/unit_icon_0025_0_4.png -> images/unit_icon_0025_0_4.png
    """
    src = clean_text(src)
    if not src:
        return None

    filename = os.path.basename(src.replace("\\", "/"))
    if not filename:
        return None

    return os.path.join(IMAGES_DIR, filename).replace("\\", "/")


def lucksack_image_url(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return f"https://static.lucksack.gg/images/monsters/{raw.lstrip('/')}"


def load_existing_image_map():
    for path in (OUTPUT_PATH,):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
            if isinstance(value, dict):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def load_lucksack_image_map():
    if not os.path.exists(LUCK_CATALOG_PATH):
        return {}
    try:
        with open(LUCK_CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(catalog, list):
        return {}
    result = {}
    for item in catalog:
        name = str(
            item.get("slabel")
            or item.get("label")
            or item.get("slug")
            or ""
        ).strip()
        image = lucksack_image_url(item.get("image"))
        if name and image:
            result.setdefault(name, image)
    return result

def parse_html_file(html_path: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, HTML_PARSER)
    image_map = {}
    missing_files = []
    skipped_non_unit_icon = 0

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/monster/" not in href.lower():
            continue

        img = a.find("img")
        if not img:
            continue

        src = clean_text(img.get("src", ""))
        if not is_unit_icon(src):
            skipped_non_unit_icon += 1
            continue

        alt = clean_text(img.get("alt", ""))

        title = clean_text(a.get("title", ""))
        if not title:
            title_node = a.find(attrs={"title": True})
            title = clean_text(title_node.get("title", "")) if title_node else ""

        raw_name = title or alt
        if not raw_name:
            continue

        base_name = extract_base_name(raw_name)
        new_path = image_src_to_images_path(src)

        if not base_name or not new_path:
            continue

        abs_check_path = os.path.normpath(new_path)
        if os.path.exists(abs_check_path):
            # 后出现的同名一般覆盖前者，通常问题不大
            image_map[base_name] = new_path
        else:
            missing_files.append({
                "monster": base_name,
                "expected_path": new_path,
                "src_in_html": src,
                "html_file": html_path.replace("\\", "/")
            })

    return image_map, missing_files, skipped_non_unit_icon

def rebuild_monster_images():
    final_map = load_existing_image_map()
    for name, image in load_lucksack_image_map().items():
        final_map.setdefault(name, image)
    all_missing = []
    total_skipped_non_unit_icon = 0
    html_count = 0

    if not os.path.exists(DATASET_DIR):
        print(f"[ERR] dataset dir not found: {DATASET_DIR}")
        return

    if not os.path.exists(IMAGES_DIR):
        print(f"[WARN] images dir not found: {IMAGES_DIR}")
        print("      你可能需要先运行 sync_images.py")
    
    for path_obj in sorted(Path(DATASET_DIR).rglob("*.html")):
        filename = path_obj.name
        html_count += 1
        html_path = str(path_obj)

        try:
            local_map, missing_files, skipped_non_unit_icon = parse_html_file(html_path)

            final_map.update(local_map)
            all_missing.extend(missing_files)
            total_skipped_non_unit_icon += skipped_non_unit_icon

            print(
                f"[OK] {filename}: "
                f"mapped={len(local_map)}, "
                f"missing={len(missing_files)}, "
                f"skipped_non_unit_icon={skipped_non_unit_icon}"
            )
        except Exception as e:
            print(f"[ERR] {filename}: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_map, f, ensure_ascii=False, indent=2)

    print("\n=== rebuild_monster_images 完成 ===")
    print(f"HTML 文件数                 : {html_count}")
    print(f"成功映射的魔灵图片数       : {len(final_map)}")
    print(f"跳过的非 unit_icon 图片数  : {total_skipped_non_unit_icon}")
    print(f"输出文件                   : {OUTPUT_PATH}")

    if all_missing:
        with open(MISSING_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_missing, f, ensure_ascii=False, indent=2)
        print(f"缺失图片报告               : {MISSING_REPORT_PATH}")
        print(f"缺失记录数                 : {len(all_missing)}")
    else:
        print("缺失图片记录               : 0")

if __name__ == "__main__":
    rebuild_monster_images()
