import json

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_jsonl(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_ban_name(picks: list):
    banned = [x["name"] for x in picks if x.get("is_banned")]
    return banned[0] if banned else None

def get_leader_name(picks: list):
    leaders = [x["name"] for x in picks if x.get("is_leader")]
    return leaders[0] if leaders else None

def get_first_pick_side(combined_sequence: list):
    """
    根据 order=1 的那条 pick 判断谁是一选
    """
    for item in combined_sequence:
        if item.get("order") == 1:
            return item["side"]
    return None

def build_combined_sequence(match: dict):
    """
    将 my_picks + enemy_picks 合并为统一序列，并按 order 排序
    """
    combined = []

    for item in match.get("my_picks", []):
        if item.get("order") is None:
            continue
        combined.append({
            "side": "me",
            "order": item["order"],
            "name": item["name"],
            "slug": item.get("slug"),
            "is_banned": item.get("is_banned", False),
            "is_leader": item.get("is_leader", False),
            "is_first_pick": item.get("is_first_pick", False),
            "title": item.get("title"),
            "alt": item.get("alt"),
        })

    for item in match.get("enemy_picks", []):
        if item.get("order") is None:
            continue
        combined.append({
            "side": "enemy",
            "order": item["order"],
            "name": item["name"],
            "slug": item.get("slug"),
            "is_banned": item.get("is_banned", False),
            "is_leader": item.get("is_leader", False),
            "is_first_pick": item.get("is_first_pick", False),
            "title": item.get("title"),
            "alt": item.get("alt"),
        })

    combined.sort(key=lambda x: x["order"])
    return combined

def validate_match_sequence(combined_sequence: list) -> bool:
    """
    校验这一场的BP顺序是否合理：
    - 必须有10手
    - order 应该是 1~10
    """
    if len(combined_sequence) != 10:
        return False

    orders = [x["order"] for x in combined_sequence]
    return orders == list(range(1, 11))

def expand_one_match(match: dict):
    combined_sequence = build_combined_sequence(match)

    if not validate_match_sequence(combined_sequence):
        return []

    first_pick_side = get_first_pick_side(combined_sequence)
    if not first_pick_side:
        return []

    my_current = []
    enemy_current = []

    my_ban = get_ban_name(match.get("my_picks", []))
    enemy_ban = get_ban_name(match.get("enemy_picks", []))
    my_leader = get_leader_name(match.get("my_picks", []))
    enemy_leader = get_leader_name(match.get("enemy_picks", []))

    player_id = match.get("page_player_id")
    match_index = match.get("match_index")
    match_id = f"{player_id}_{match_index}"

    samples = []

    for step in combined_sequence:
        sample = {
            "sample_id": f"{match_id}_{step['order']}",
            "match_id": match_id,
            "player_id": player_id,
            "match_index": match_index,
            "result": match.get("result"),
            "time_ago": match.get("time_ago"),
            "self_name": match.get("self", {}).get("name"),
            "self_rating": match.get("self", {}).get("rating"),
            "opponent_name": match.get("opponent", {}).get("name"),
            "opponent_rating": match.get("opponent", {}).get("rating"),
            "turn": step["order"],
            "first_pick_side": first_pick_side,
            "current_side": step["side"],
            "my_picks": my_current.copy(),
            "enemy_picks": enemy_current.copy(),
            "my_ban": my_ban,
            "enemy_ban": enemy_ban,
            "my_leader": my_leader,
            "enemy_leader": enemy_leader,
            "next_pick": step["name"],
            "next_pick_slug": step.get("slug")
        }

        samples.append(sample)

        # 更新状态
        if step["side"] == "me":
            my_current.append(step["name"])
        else:
            enemy_current.append(step["name"])

    return samples

def expand_all_matches(matches_raw: list):
    all_samples = []
    valid_match_count = 0
    skipped_match_count = 0

    for match in matches_raw:
        samples = expand_one_match(match)
        if samples:
            all_samples.extend(samples)
            valid_match_count += 1
        else:
            skipped_match_count += 1

    return all_samples, valid_match_count, skipped_match_count

def build_me_turn_only_samples(all_samples: list):
    """
    如果你后面只想训练“轮到我方时该选什么”，可以保留 current_side == me 的样本
    """
    return [x for x in all_samples if x["current_side"] == "me"]

def main():
    input_path = "json//matches_raw.json"
    output_all_json = "json//bp_training_samples_from_raw.json"
    output_all_jsonl = "json//bp_training_samples_from_raw.jsonl"
    output_me_only_json = "json//bp_training_samples_me_only.json"
    output_me_only_jsonl = "json//bp_training_samples_me_only.jsonl"

    matches_raw = load_json(input_path)

    all_samples, valid_match_count, skipped_match_count = expand_all_matches(matches_raw)
    me_only_samples = build_me_turn_only_samples(all_samples)

    save_json(all_samples, output_all_json)
    save_jsonl(all_samples, output_all_jsonl)
    save_json(me_only_samples, output_me_only_json)
    save_jsonl(me_only_samples, output_me_only_jsonl)

    print(f"原始比赛数: {len(matches_raw)}")
    print(f"有效比赛数: {valid_match_count}")
    print(f"跳过比赛数: {skipped_match_count}")
    print(f"全部训练样本数: {len(all_samples)}")
    print(f"仅我方回合样本数: {len(me_only_samples)}")

    print(f"\n已保存:")
    print(f"  {output_all_json}")
    print(f"  {output_all_jsonl}")
    print(f"  {output_me_only_json}")
    print(f"  {output_me_only_jsonl}")

    print("\n前3条样本预览:")
    for x in all_samples[:3]:
        print(json.dumps(x, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()