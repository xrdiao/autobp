import os
import re
import json
from bs4 import BeautifulSoup
from collections import defaultdict

ATTRS = {"Fire", "Water", "Wind", "Light", "Dark"}

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def extract_slug(href: str):
    if not href:
        return None
    m = re.search(r"/monster/([^/?#]+)", href)
    return m.group(1).strip() if m else None

def extract_base_name(text: str):
    text = clean_text(text)
    text = re.sub(r"\s*\(Leader\)", "", text, flags=re.I)
    text = re.sub(r"\s*\(Banned\)", "", text, flags=re.I)

    m = re.match(r"^(.*?)\s*\(", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def parse_attr_and_family(text: str):
    """
    例如:
    Oliver (Wind Sky Surfer) -> attr=Wind, family=Sky Surfer
    Josephine (Water Paladin) -> attr=Water, family=Paladin
    """
    text = clean_text(text)
    text = re.sub(r"\s*\(Leader\)", "", text, flags=re.I)
    text = re.sub(r"\s*\(Banned\)", "", text, flags=re.I)

    m = re.search(r"\(([^()]*)\)", text)
    if not m:
        return None, None

    inside = clean_text(m.group(1))
    parts = inside.split(" ", 1)
    if len(parts) == 2 and parts[0] in ATTRS:
        return parts[0], parts[1].strip()
    return None, inside

def resolve_image_path(src: str, html_path: str):
    """
    把 html 里的相对图片路径，解析成相对项目根目录可访问的路径
    """
    src = clean_text(src)
    if not src:
        return ""

    if src.startswith("http://") or src.startswith("https://"):
        return src

    html_dir = os.path.dirname(html_path)
    abs_path = os.path.normpath(os.path.join(html_dir, src))
    return abs_path.replace("\\", "/")

def parse_monsters_from_html(html_path: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/monster/" not in href:
            continue

        slug = extract_slug(href)
        if not slug:
            continue

        img = a.find("img")
        if not img:
            continue

        alt = clean_text(img.get("alt", ""))
        src = clean_text(img.get("src", ""))

        title_node = a.find(attrs={"title": True})
        title = clean_text(title_node.get("title", "")) if title_node else ""

        raw_name = title or alt or slug
        base_name = extract_base_name(raw_name)

        attr, family = parse_attr_and_family(title or alt)
        resolved_image = resolve_image_path(src, html_path)

        results.append({
            "slug": slug,
            "raw_name": raw_name,
            "base_name": base_name,
            "alt": alt,
            "title": title,
            "image": resolved_image,
            "attribute": attr,
            "family": family,
            "source_html": os.path.basename(html_path)
        })

    return results

def build_catalog(html_dir: str, output_path="monster_catalog.json"):
    all_entries = []

    for filename in os.listdir(html_dir):
        if not filename.endswith(".html"):
            continue
        path = os.path.join(html_dir, filename)
        try:
            entries = parse_monsters_from_html(path)
            all_entries.extend(entries)
            print(f"[OK] {filename}: {len(entries)}")
        except Exception as e:
            print(f"[ERR] {filename}: {e}")

    grouped = defaultdict(lambda: {
        "images": set(),
        "slugs": set(),
        "raw_names": set(),
        "alts": set(),
        "titles": set(),
        "attributes": set(),
        "families": set(),
        "sources": set(),
    })

    for item in all_entries:
        key = item["base_name"]
        grouped[key]["images"].add(item["image"])
        grouped[key]["slugs"].add(item["slug"])
        grouped[key]["raw_names"].add(item["raw_name"])
        grouped[key]["alts"].add(item["alt"])
        grouped[key]["titles"].add(item["title"])
        if item["attribute"]:
            grouped[key]["attributes"].add(item["attribute"])
        if item["family"]:
            grouped[key]["families"].add(item["family"])
        grouped[key]["sources"].add(item["source_html"])

    catalog = []
    for base_name, info in grouped.items():
        catalog.append({
            "base_name": base_name,
            "images": sorted(x for x in info["images"] if x),
            "slugs": sorted(x for x in info["slugs"] if x),
            "raw_names": sorted(x for x in info["raw_names"] if x),
            "alts": sorted(x for x in info["alts"] if x),
            "titles": sorted(x for x in info["titles"] if x),
            "attributes": sorted(x for x in info["attributes"] if x),
            "families": sorted(x for x in info["families"] if x),
            "sources": sorted(x for x in info["sources"] if x),
        })

    catalog.sort(key=lambda x: x["base_name"].lower())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Monster count: {len(catalog)}")

if __name__ == "__main__":
    build_catalog("dataset", "json//monster_catalog.json")