import os
import json
from extract_match_bp import process_html_file

INPUT_DIR = "D:\\jupyter\\auto_bp\\dataset"
ALL_RAW = []
ALL_DATASET = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".html"):
        continue

    path = os.path.join(INPUT_DIR, filename)

    try:
        raw_matches, dataset_matches = process_html_file(path)
        ALL_RAW.extend(raw_matches)
        ALL_DATASET.extend(dataset_matches)
        print(f"[OK] {filename}: {len(dataset_matches)} matches")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

with open("json//matches_raw.json", "w", encoding="utf-8") as f:
    json.dump(ALL_RAW, f, ensure_ascii=False, indent=2)

with open("json//bp_dataset.json", "w", encoding="utf-8") as f:
    json.dump(ALL_DATASET, f, ensure_ascii=False, indent=2)

print(f"\nTotal raw matches: {len(ALL_RAW)}")
print(f"Total dataset matches: {len(ALL_DATASET)}")