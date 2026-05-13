import hashlib
import json
import os
import re
import shutil
from pathlib import Path

DATASET_DIR = "dataset"
IMAGES_DIR = "images"
CONFLICT_REPORT = "json//image_conflicts.json"
SYNC_REPORT = "json//image_sync_report.json"

UNIT_ICON_PATTERN = re.compile(r"^unit_icon.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def file_hash(path: str, chunk_size=8192):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_unit_icon_file(filename: str):
    return bool(UNIT_ICON_PATTERN.match(filename))


def find_resource_dirs(dataset_dir: str):
    root = Path(dataset_dir)
    if not root.exists():
        return []

    result = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.endswith("_files"):
            result.append(str(path))
    return sorted(result)


def sync_images():
    ensure_dir(IMAGES_DIR)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERR] dataset dir not found: {DATASET_DIR}")
        return

    resource_dirs = find_resource_dirs(DATASET_DIR)

    copied = 0
    skipped_same = 0
    conflicts = []
    scanned_files = 0
    skipped_non_unit_icon = 0
    detail_records = []

    for resource_dir in resource_dirs:
        for filename in os.listdir(resource_dir):
            src_path = os.path.join(resource_dir, filename)

            if not os.path.isfile(src_path):
                continue

            scanned_files += 1

            if not is_unit_icon_file(filename):
                skipped_non_unit_icon += 1
                continue

            dst_path = os.path.join(IMAGES_DIR, filename)

            record = {
                "filename": filename,
                "src_path": src_path.replace("\\", "/"),
                "dst_path": dst_path.replace("\\", "/"),
                "status": None,
            }

            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
                record["status"] = "copied"
            else:
                src_hash = file_hash(src_path)
                dst_hash = file_hash(dst_path)

                if src_hash == dst_hash:
                    skipped_same += 1
                    record["status"] = "skipped_same"
                else:
                    conflicts.append(
                        {
                            "filename": filename,
                            "src_path": src_path.replace("\\", "/"),
                            "dst_path": dst_path.replace("\\", "/"),
                            "src_hash": src_hash,
                            "dst_hash": dst_hash,
                        }
                    )
                    record["status"] = "conflict"

            detail_records.append(record)

    report = {
        "dataset_dir": DATASET_DIR,
        "images_dir": IMAGES_DIR,
        "resource_dir_count": len(resource_dirs),
        "scanned_files": scanned_files,
        "skipped_non_unit_icon": skipped_non_unit_icon,
        "copied": copied,
        "skipped_same": skipped_same,
        "conflict_count": len(conflicts),
        "resource_dirs": [x.replace("\\", "/") for x in resource_dirs],
        "details": detail_records[:1000],
    }

    with open(SYNC_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if conflicts:
        with open(CONFLICT_REPORT, "w", encoding="utf-8") as f:
            json.dump(conflicts, f, ensure_ascii=False, indent=2)

    print("=== unit_icon sync finished ===")
    print(f"resource dirs   : {len(resource_dirs)}")
    print(f"scanned files   : {scanned_files}")
    print(f"skipped non icon: {skipped_non_unit_icon}")
    print(f"copied files    : {copied}")
    print(f"same files      : {skipped_same}")
    print(f"conflicts       : {len(conflicts)}")
    print(f"sync report     : {SYNC_REPORT}")
    if conflicts:
        print(f"conflict report : {CONFLICT_REPORT}")


if __name__ == "__main__":
    sync_images()
