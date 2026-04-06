import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_name(name, alias_map):
    if not name:
        return name
    return alias_map.get(name, name)

def normalize_list(names, alias_map):
    return [normalize_name(x, alias_map) for x in names]

def main():
    samples = load_json("json//bp_training_samples_me_only.json")
    alias_map = load_json("json//monster_alias_map.json")

    normalized = []

    for s in samples:
        item = dict(s)
        item["my_picks"] = normalize_list(item.get("my_picks", []), alias_map)
        item["enemy_picks"] = normalize_list(item.get("enemy_picks", []), alias_map)
        item["next_pick"] = normalize_name(item.get("next_pick"), alias_map)
        item["my_ban"] = normalize_name(item.get("my_ban"), alias_map)
        item["enemy_ban"] = normalize_name(item.get("enemy_ban"), alias_map)
        item["my_leader"] = normalize_name(item.get("my_leader"), alias_map)
        item["enemy_leader"] = normalize_name(item.get("enemy_leader"), alias_map)
        normalized.append(item)

    save_json(normalized, "json//bp_training_samples_me_only_normalized.json")
    print("Saved bp_training_samples_me_only_normalized.json")

if __name__ == "__main__":
    main()