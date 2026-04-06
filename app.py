import json
import os
import math
import base64
from collections import Counter, defaultdict
from itertools import combinations
import streamlit as st
from bp_recommender_from_samples import (
    load_samples,
    BPRecommenderFromSamples,
    infer_turn_from_state
)

SAMPLES_PATH = "json/bp_training_samples_me_only_normalized.json"
MY_BOX_PATH = "json/my_box.json"
IMAGE_MAP_PATH = "json/monster_images.json"
ATTR_MAP_PATH = "json/monster_attributes.json"
FAMILY_MAP_PATH = "json/monster_families.json"

@st.cache_data
def load_data():
    samples = load_samples(SAMPLES_PATH)
    recommender = BPRecommenderFromSamples(samples)
    pick_freq = build_pick_frequency(samples)
    coverage_stats = build_sample_coverage_stats(samples)
    return samples, recommender, pick_freq, coverage_stats

@st.cache_data
def load_json_file(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

@st.cache_data
def image_file_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def build_pick_frequency(samples):
    freq = Counter()
    for s in samples:
        for m in s.get("my_picks", []):
            if m:
                freq[m] += 1
        for m in s.get("enemy_picks", []):
            if m:
                freq[m] += 1
        nxt = s.get("next_pick")
        if nxt:
            freq[nxt] += 1
    return freq

@st.cache_data
def build_sample_coverage_stats(samples):
    single_freq = Counter()
    pair_freq = Counter()
    lineup_freq = Counter()

    for s in samples:
        for side_key in ["my_picks", "enemy_picks"]:
            picks = [x for x in s.get(side_key, []) if x]
            uniq = sorted(set(picks))

            for m in uniq:
                single_freq[m] += 1

            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    pair = tuple(sorted([uniq[i], uniq[j]]))
                    pair_freq[pair] += 1

            if len(uniq) >= 3:
                lineup_freq[tuple(uniq)] += 1

    return {
        "top_singles": [m for m, _ in single_freq.most_common(50)],
        "top_pairs": [pair for pair, _ in pair_freq.most_common(100)],
        "top_lineups": [lineup for lineup, _ in lineup_freq.most_common(80)]
    }

def sort_monsters_by_pick_frequency(monsters, pick_freq):
    return sorted(monsters, key=lambda m: (-pick_freq.get(m, 0), m))

def resolve_image_for_streamlit(path: str):
    if not path:
        return None

    path = path.strip().replace("/", os.sep).replace("\\", os.sep)

    if os.path.isabs(path):
        return path if os.path.exists(path) else None

    if os.path.exists(path):
        return os.path.normpath(path)

    filename = os.path.basename(path)
    candidates = [
        os.path.join(os.getcwd(), path),
        os.path.join(os.getcwd(), "images", filename),
        os.path.join("images", filename),
    ]

    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None

def init_state(initial_my_box=None):
    defaults = {
        "my_picks": [],
        "enemy_picks": [],
        "banned_monsters": [],
        "final_bans": [],
        "search_text": "",
        "last_action": "",
        "pick_plans": [],
        "ban_plans": [],
        "enemy_predictions": [],
        "phase_hint": "等待操作",
        "my_box_state": initial_my_box[:] if isinstance(initial_my_box, list) else [],
        "box_analysis_result": None,
        "coverage_analysis_result": None,
        "best_addition_result": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def invalidate_box_analysis():
    st.session_state.box_analysis_result = None
    st.session_state.coverage_analysis_result = None
    st.session_state.best_addition_result = None

def update_phase_hint(first_pick_side):
    my_count = len(st.session_state.my_picks)
    enemy_count = len(st.session_state.enemy_picks)

    if my_count == 5 and enemy_count == 5:
        if st.session_state.final_bans:
            st.session_state.phase_hint = "已选择最终 Ban，BP 完成"
        else:
            st.session_state.phase_hint = "双方选人完成，可进行 Ban 胜率预测"
        return

    turn = infer_turn_from_state(
        st.session_state.my_picks,
        st.session_state.enemy_picks,
        first_pick_side
    )

    if turn is None:
        st.session_state.phase_hint = "当前为对方操作阶段"
    else:
        st.session_state.phase_hint = f"当前为我方选人阶段（第 {turn} 手）"

def clear_recommendations():
    st.session_state.pick_plans = []
    st.session_state.ban_plans = []
    st.session_state.enemy_predictions = []

def remove_from_all(monster):
    if monster in st.session_state.my_picks:
        st.session_state.my_picks.remove(monster)
    if monster in st.session_state.enemy_picks:
        st.session_state.enemy_picks.remove(monster)
    if monster in st.session_state.banned_monsters:
        st.session_state.banned_monsters.remove(monster)

def add_unique(lst_name, monster, max_len=5):
    remove_from_all(monster)
    lst = st.session_state[lst_name]
    if len(lst) < max_len:
        lst.append(monster)
        clear_recommendations()
        st.session_state.last_action = f"已将 {monster} 加入 {lst_name}"
    else:
        st.session_state.last_action = f"{lst_name} 已满，无法加入 {monster}"

def add_ban(monster):
    remove_from_all(monster)
    if monster not in st.session_state.banned_monsters:
        st.session_state.banned_monsters.append(monster)
        clear_recommendations()
        st.session_state.last_action = f"已将 {monster} 加入 ban"

def clear_all():
    st.session_state.my_picks = []
    st.session_state.enemy_picks = []
    st.session_state.banned_monsters = []
    st.session_state.final_bans = []
    st.session_state.pick_plans = []
    st.session_state.ban_plans = []
    st.session_state.enemy_predictions = []
    st.session_state.last_action = "已清空当前 BP 状态"
    st.session_state.phase_hint = "等待操作"

def remove_one(monster):
    remove_from_all(monster)
    if monster in st.session_state.final_bans:
        st.session_state.final_bans.remove(monster)
    clear_recommendations()
    st.session_state.last_action = f"已移除 {monster}"

def apply_pick_plan(plan, first_pick_side):
    plan = [x for x in plan if x]
    unique_plan = []
    for m in plan:
        if m not in unique_plan:
            unique_plan.append(m)

    current = st.session_state.my_picks[:]
    remain_slots = 5 - len(current)
    to_add = [m for m in unique_plan if m not in current]

    if len(to_add) > remain_slots:
        st.session_state.last_action = f"无法采用方案，当前我方剩余位置不足（剩余 {remain_slots}）"
        return

    for monster in to_add:
        remove_from_all(monster)
        st.session_state.my_picks.append(monster)

    st.session_state.pick_plans = []
    st.session_state.ban_plans = []
    st.session_state.enemy_predictions = []

    update_phase_hint(first_pick_side)

    if len(st.session_state.my_picks) == 5 and len(st.session_state.enemy_picks) == 5:
        st.session_state.last_action = f"已采用推荐方案: {' + '.join(to_add)}。双方选人完成，可生成 Ban 推荐。"
    else:
        turn = infer_turn_from_state(
            st.session_state.my_picks,
            st.session_state.enemy_picks,
            first_pick_side
        )
        if turn is not None:
            st.session_state.last_action = f"已采用推荐方案: {' + '.join(to_add)}。已进入下一轮我方选人阶段。"
        else:
            st.session_state.last_action = f"已采用推荐方案: {' + '.join(to_add)}。当前轮到对方操作。"

def apply_ban_plan(monster, win_rate=None):
    if not monster:
        return

    st.session_state.final_bans = [monster]
    st.session_state.ban_plans = []

    if win_rate is not None:
        st.session_state.last_action = f"已采用 Ban 方案: {monster}，预测胜率 {win_rate}"
    else:
        st.session_state.last_action = f"已采用 Ban 方案: {monster}"

    st.session_state.phase_hint = "已选择最终 Ban，BP 完成"

def toggle_my_box(monster):
    if monster in st.session_state.my_box_state:
        st.session_state.my_box_state.remove(monster)
        st.session_state.last_action = f"已从我的 Box 移除 {monster}"
    else:
        st.session_state.my_box_state.append(monster)
        st.session_state.last_action = f"已加入我的 Box: {monster}"
    invalidate_box_analysis()

def save_my_box():
    sorted_box = sorted(set(st.session_state.my_box_state))
    with open(MY_BOX_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_box, f, ensure_ascii=False, indent=2)
    st.session_state.last_action = f"已保存我的 Box，共 {len(sorted_box)} 个魔灵"

def clear_my_box():
    st.session_state.my_box_state = []
    st.session_state.last_action = "已清空我的 Box（尚未保存到文件）"
    invalidate_box_analysis()

def bulk_add_to_box(monsters):
    before = len(st.session_state.my_box_state)
    merged = set(st.session_state.my_box_state)
    merged.update(monsters)
    st.session_state.my_box_state = sorted(merged)
    after = len(st.session_state.my_box_state)
    st.session_state.last_action = f"已批量加入 {after - before} 个魔灵到我的 Box"
    invalidate_box_analysis()

def bulk_remove_from_box(monsters):
    before = len(st.session_state.my_box_state)
    remove_set = set(monsters)
    st.session_state.my_box_state = [m for m in st.session_state.my_box_state if m not in remove_set]
    after = len(st.session_state.my_box_state)
    st.session_state.last_action = f"已批量移除 {before - after} 个魔灵出我的 Box"
    invalidate_box_analysis()

def add_monsters_to_box_temp(monsters):
    before = len(st.session_state.my_box_state)
    merged = set(st.session_state.my_box_state)
    merged.update([m for m in monsters if m])
    st.session_state.my_box_state = sorted(merged)
    after = len(st.session_state.my_box_state)
    st.session_state.last_action = f"已临时加入 {after - before} 个建议魔灵到当前 Box"
    invalidate_box_analysis()

def add_monsters_to_box_and_save(monsters):
    add_monsters_to_box_temp(monsters)
    save_my_box()
    st.session_state.last_action = f"已加入并保存建议魔灵：{', '.join(monsters)}"

def filter_monsters(
    monster_pool,
    search_text,
    attr_filter,
    family_filter,
    only_owned=False,
    my_box=None,
    attr_map=None,
    family_map=None,
    only_with_image=False,
    image_map=None
):
    result = monster_pool

    if only_owned and my_box:
        result = [m for m in result if m in my_box]

    if search_text:
        s = search_text.lower().strip()
        result = [m for m in result if s in m.lower()]

    if attr_filter != "全部":
        result = [m for m in result if attr_map.get(m) == attr_filter]

    if family_filter != "全部":
        result = [m for m in result if family_map.get(m) == family_filter]

    if only_with_image and image_map:
        result = [m for m in result if resolve_image_for_streamlit(image_map.get(m))]

    return result

def render_selected_monsters(title, monsters, image_map, key_prefix):
    st.markdown(f"### {title}")
    if not monsters:
        st.caption("无")
        return

    cols = st.columns(min(5, max(1, len(monsters))))
    for i, monster in enumerate(monsters):
        with cols[i % len(cols)]:
            image_path = resolve_image_for_streamlit(image_map.get(monster))
            if image_path:
                st.image(image_path, caption=monster, use_container_width=True)
            else:
                st.write(f"🧩 {monster}")

            if st.button("移除", key=f"{key_prefix}_{monster}_{i}"):
                remove_one(monster)

def render_current_box_top(monsters, image_map, pick_freq):
    with st.expander(f"当前 Box 已有魔灵（{len(monsters)}）", expanded=False):
        if not monsters:
            st.caption("当前 Box 为空")
            return

        sorted_monsters = sort_monsters_by_pick_frequency(monsters, pick_freq)
        columns_per_row = 10
        rows = math.ceil(len(sorted_monsters) / columns_per_row)

        for row in range(rows):
            cols = st.columns(columns_per_row)
            row_items = sorted_monsters[row * columns_per_row:(row + 1) * columns_per_row]

            for col_idx, (col, monster) in enumerate(zip(cols, row_items)):
                with col:
                    image_path = resolve_image_for_streamlit(image_map.get(monster))
                    if image_path:
                        st.image(image_path, caption=monster, use_container_width=True)
                    else:
                        st.caption(monster)

                    if st.button("移出", key=f"top_box_remove_{monster}_{row}_{col_idx}"):
                        toggle_my_box(monster)

def get_monster_style_tags(monster, family_map):
    family = (family_map.get(monster) or "").lower()
    name = monster.lower()

    tags = []

    keyword_rules = [
        (["sky surfer"], "先手压制"),
        (["oracle"], "控场"),
        (["paladin"], "耐久"),
        (["totemist"], "续航"),
        (["dragon knight"], "爆发"),
        (["chimera"], "爆发"),
        (["monkey king"], "反打"),
        (["pioneer"], "功能"),
        (["beast rider"], "机动"),
        (["fairy king"], "功能"),
        (["occult girl"], "控场"),
        (["desert queen"], "功能"),
        (["druid"], "续航"),
        (["archangel"], "续航"),
        (["battle angel"], "压制"),
    ]

    for keywords, tag in keyword_rules:
        if any(k in family for k in keywords):
            tags.append(tag)

    name_rules = [
        (["racuni"], "解控"),
        (["velajuel"], "解控"),
        (["josephine"], "反控"),
        (["oliver"], "先手压制"),
        (["sonia"], "爆发"),
        (["masha"], "压制"),
        (["vanessa"], "功能"),
        (["nana"], "续航"),
        (["haegang"], "反控"),
        (["giou"], "控场"),
        (["seara"], "爆发"),
    ]

    for keywords, tag in name_rules:
        if any(k in name for k in keywords):
            tags.append(tag)

    return tags

def get_plan_style_tags(plan, family_map):
    counter = Counter()
    for monster in plan:
        tags = get_monster_style_tags(monster, family_map)
        for t in tags:
            counter[t] += 1

    if not counter:
        return ["均衡"]

    return [tag for tag, _ in counter.most_common(3)]

def analyze_box(box, monster_pool, attr_map, family_map, samples):
    box = list(sorted(set(box)))
    if not box:
        return {
            "score": 0,
            "summary": ["当前 Box 为空，无法进行阵容自洽分析。"],
            "add_suggestions": [],
            "remove_suggestions": [],
            "attr_counter": {},
            "style_counter": {}
        }

    attr_counter = Counter(attr_map.get(m, "未知") for m in box)

    style_counter = Counter()
    for m in box:
        for tag in get_monster_style_tags(m, family_map):
            style_counter[tag] += 1

    score = 50

    if len(box) >= 20:
        score += 10
    elif len(box) >= 10:
        score += 5

    unique_attrs = len([k for k, v in attr_counter.items() if v > 0 and k != "未知"])
    score += min(unique_attrs * 5, 25)

    core_tags = ["先手压制", "控场", "续航", "爆发", "解控", "耐久", "反打", "功能"]
    covered_tags = sum(1 for t in core_tags if style_counter.get(t, 0) > 0)
    score += min(covered_tags * 2, 15)

    score = min(score, 100)

    summary = []
    summary.append(f"当前 Box 共 {len(box)} 个魔灵。")
    summary.append(f"属性覆盖：{', '.join([f'{k}:{v}' for k, v in attr_counter.items()]) if attr_counter else '无'}")
    if style_counter:
        summary.append(f"风格覆盖：{', '.join([f'{k}:{v}' for k, v in style_counter.most_common(8)])}")
    else:
        summary.append("风格覆盖：暂未识别出明显标签，整体偏均衡或数据不足。")

    sample_freq = Counter()
    for s in samples:
        for m in s.get("my_picks", []):
            sample_freq[m] += 1
        for m in s.get("enemy_picks", []):
            sample_freq[m] += 1

    missing_tags = [t for t in core_tags if style_counter.get(t, 0) == 0]

    candidate_adds = []
    for m, freq in sample_freq.most_common():
        if m in box:
            continue
        m_tags = get_monster_style_tags(m, family_map)
        bonus = sum(3 for t in m_tags if t in missing_tags)
        attr = attr_map.get(m, "未知")
        if attr_counter.get(attr, 0) == 0 and attr != "未知":
            bonus += 2
        total_score = freq + bonus
        candidate_adds.append((m, total_score, freq, m_tags, attr))

    add_suggestions = []
    for m, total_score, freq, tags, attr in candidate_adds[:10]:
        reason = []
        if tags:
            reason.append(f"可补足风格：{' / '.join(tags[:3])}")
        if attr_counter.get(attr, 0) == 0 and attr != "未知":
            reason.append(f"可补属性短板：{attr}")
        reason.append(f"样本出现频次高：{freq}")
        add_suggestions.append({
            "monster": m,
            "reason": "；".join(reason)
        })

    box_freq = []
    for m in box:
        freq = sample_freq.get(m, 0)
        tags = get_monster_style_tags(m, family_map)
        redundant = 0
        for t in tags:
            if style_counter.get(t, 0) >= 3:
                redundant += 1
        attr = attr_map.get(m, "未知")
        attr_is_redundant = attr_counter.get(attr, 0) >= 4
        remove_score = (3 - min(freq, 3)) + redundant + (1 if attr_is_redundant else 0)
        box_freq.append((m, remove_score, freq, tags, attr, attr_is_redundant))

    box_freq.sort(key=lambda x: x[1], reverse=True)

    remove_suggestions = []
    for m, remove_score, freq, tags, attr, attr_is_redundant in box_freq[:8]:
        if remove_score < 2:
            continue
        reason = []
        if freq <= 1:
            reason.append("历史样本中出现较少")
        if tags:
            repeated_tags = [t for t in tags if style_counter.get(t, 0) >= 3]
            if repeated_tags:
                reason.append(f"与现有 Box 风格重复：{' / '.join(repeated_tags[:3])}")
        if attr_is_redundant and attr != "未知":
            reason.append(f"{attr} 属性数量较多")
        if reason:
            remove_suggestions.append({
                "monster": m,
                "reason": "；".join(reason)
            })

    return {
        "score": score,
        "summary": summary,
        "add_suggestions": add_suggestions[:5],
        "remove_suggestions": remove_suggestions[:5],
        "attr_counter": dict(attr_counter),
        "style_counter": dict(style_counter)
    }

def build_box_coverage_analysis(box, coverage_stats):
    box_set = set(box or [])
    if not box_set:
        return {
            "single_cover_rate": 0.0,
            "pair_cover_rate": 0.0,
            "lineup_cover_rate": 0.0,
            "top_missing_adds": [],
            "low_value_owned": [],
            "single_total": 0,
            "pair_total": 0,
            "lineup_total": 0,
            "top_singles": [],
            "top_pairs": [],
            "top_lineups": []
        }

    top_singles = coverage_stats["top_singles"]
    top_pairs = coverage_stats["top_pairs"]
    top_lineups = coverage_stats["top_lineups"]

    single_covered = sum(1 for m in top_singles if m in box_set)
    pair_covered = sum(1 for pair in top_pairs if pair[0] in box_set and pair[1] in box_set)

    lineup_covered = 0
    lineup_missing_map = defaultdict(int)
    for lineup in top_lineups:
        owned_count = sum(1 for m in lineup if m in box_set)
        ratio = owned_count / len(lineup)
        if ratio >= 0.8:
            lineup_covered += 1
        else:
            missing = [m for m in lineup if m not in box_set]
            if 1 <= len(missing) <= 2:
                for m in missing:
                    lineup_missing_map[m] += 1

    single_total = len(top_singles)
    pair_total = len(top_pairs)
    lineup_total = len(top_lineups)

    single_cover_rate = round(single_covered / single_total, 4) if single_total else 0.0
    pair_cover_rate = round(pair_covered / pair_total, 4) if pair_total else 0.0
    lineup_cover_rate = round(lineup_covered / lineup_total, 4) if lineup_total else 0.0

    add_scores = []
    for m, lineup_gain in lineup_missing_map.items():
        single_gain = 1 if m in top_singles else 0
        pair_gain = sum(
            1 for pair in top_pairs
            if m in pair and all(x in box_set or x == m for x in pair)
        )
        total_gain_score = single_gain * 2 + pair_gain * 3 + lineup_gain * 4
        add_scores.append({
            "monster": m,
            "single_gain": single_gain,
            "pair_gain": pair_gain,
            "lineup_gain": lineup_gain,
            "total_gain_score": total_gain_score
        })

    add_scores.sort(key=lambda x: x["total_gain_score"], reverse=True)

    owned_scores = []
    for m in box_set:
        single_contrib = 1 if m in top_singles else 0
        pair_contrib = sum(1 for pair in top_pairs if m in pair and pair[0] in box_set and pair[1] in box_set)
        lineup_contrib = sum(
            1 for lineup in top_lineups
            if m in lineup and sum(1 for x in lineup if x in box_set) / len(lineup) >= 0.8
        )
        total_contrib = single_contrib * 2 + pair_contrib * 3 + lineup_contrib * 4
        owned_scores.append({
            "monster": m,
            "single_contrib": single_contrib,
            "pair_contrib": pair_contrib,
            "lineup_contrib": lineup_contrib,
            "total_contrib": total_contrib
        })

    owned_scores.sort(key=lambda x: x["total_contrib"])

    return {
        "single_cover_rate": single_cover_rate,
        "pair_cover_rate": pair_cover_rate,
        "lineup_cover_rate": lineup_cover_rate,
        "top_missing_adds": add_scores[:8],
        "low_value_owned": owned_scores[:8],
        "single_total": single_total,
        "pair_total": pair_total,
        "lineup_total": lineup_total,
        "top_singles": top_singles,
        "top_pairs": top_pairs,
        "top_lineups": top_lineups
    }

def evaluate_box_coverage(box_set, top_singles, top_pairs, top_lineups):
    single_total = len(top_singles)
    pair_total = len(top_pairs)
    lineup_total = len(top_lineups)

    single_covered = sum(1 for m in top_singles if m in box_set)
    pair_covered = sum(1 for pair in top_pairs if pair[0] in box_set and pair[1] in box_set)
    lineup_covered = sum(
        1 for lineup in top_lineups
        if (sum(1 for m in lineup if m in box_set) / len(lineup)) >= 0.8
    ) if lineup_total else 0

    single_rate = single_covered / single_total if single_total else 0.0
    pair_rate = pair_covered / pair_total if pair_total else 0.0
    lineup_rate = lineup_covered / lineup_total if lineup_total else 0.0

    overall_score = round(single_rate * 0.25 + pair_rate * 0.35 + lineup_rate * 0.40, 4)

    return {
        "single_rate": single_rate,
        "pair_rate": pair_rate,
        "lineup_rate": lineup_rate,
        "overall_score": overall_score
    }

def find_best_additions(box, coverage_stats, top_n_candidates=20):
    analysis = build_box_coverage_analysis(box, coverage_stats)
    current_box = set(box or [])

    top_singles = analysis["top_singles"]
    top_pairs = analysis["top_pairs"]
    top_lineups = analysis["top_lineups"]

    if not top_singles and not top_pairs and not top_lineups:
        return {"best_one": [], "best_three": [], "base_eval": {}}

    candidate_pool = []
    for item in analysis["top_missing_adds"]:
        if item["monster"] not in current_box:
            candidate_pool.append(item["monster"])

    for m in top_singles:
        if m not in current_box and m not in candidate_pool:
            candidate_pool.append(m)
        if len(candidate_pool) >= top_n_candidates:
            break

    base_eval = evaluate_box_coverage(current_box, top_singles, top_pairs, top_lineups)

    best_one = []
    for m in candidate_pool:
        new_box = set(current_box)
        new_box.add(m)
        ev = evaluate_box_coverage(new_box, top_singles, top_pairs, top_lineups)
        best_one.append({
            "monsters": [m],
            "overall_score": ev["overall_score"],
            "gain": round(ev["overall_score"] - base_eval["overall_score"], 4),
            "single_gain": round(ev["single_rate"] - base_eval["single_rate"], 4),
            "pair_gain": round(ev["pair_rate"] - base_eval["pair_rate"], 4),
            "lineup_gain": round(ev["lineup_rate"] - base_eval["lineup_rate"], 4)
        })

    best_one.sort(key=lambda x: x["gain"], reverse=True)

    best_three = []
    combo_pool = candidate_pool[:8]
    for combo in combinations(combo_pool, 3):
        new_box = set(current_box)
        new_box.update(combo)
        ev = evaluate_box_coverage(new_box, top_singles, top_pairs, top_lineups)
        best_three.append({
            "monsters": list(combo),
            "overall_score": ev["overall_score"],
            "gain": round(ev["overall_score"] - base_eval["overall_score"], 4),
            "single_gain": round(ev["single_rate"] - base_eval["single_rate"], 4),
            "pair_gain": round(ev["pair_rate"] - base_eval["pair_rate"], 4),
            "lineup_gain": round(ev["lineup_rate"] - base_eval["lineup_rate"], 4)
        })

    best_three.sort(key=lambda x: x["gain"], reverse=True)

    return {
        "base_eval": base_eval,
        "best_one": best_one[:5],
        "best_three": best_three[:5]
    }

def render_suggestion_action_buttons(monsters, key_prefix):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("临时加入Box", key=f"{key_prefix}_temp"):
            add_monsters_to_box_temp(monsters)
    with col2:
        if st.button("加入并保存", key=f"{key_prefix}_save"):
            add_monsters_to_box_and_save(monsters)

def render_box_coverage_analysis_result(analysis, image_map):
    st.markdown("### 基于训练样本的 Box 组合覆盖分析")

    c1, c2, c3 = st.columns(3)
    c1.metric("高频单卡覆盖率", f"{analysis['single_cover_rate'] * 100:.1f}%")
    c2.metric("高频双卡覆盖率", f"{analysis['pair_cover_rate'] * 100:.1f}%")
    c3.metric("高频阵容近似覆盖率", f"{analysis['lineup_cover_rate'] * 100:.1f}%")

    st.caption(
        f"统计范围：高频单卡 {analysis['single_total']} 个，高频双卡 {analysis['pair_total']} 组，高频阵容 {analysis['lineup_total']} 组"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### 建议优先补的关键拼图")
        if analysis["top_missing_adds"]:
            for idx, item in enumerate(analysis["top_missing_adds"], start=1):
                monster = item["monster"]
                image_path = resolve_image_for_streamlit(image_map.get(monster))
                row = st.columns([1, 3])
                with row[0]:
                    if image_path:
                        st.image(image_path, use_container_width=True)
                with row[1]:
                    st.write(f"**{idx}. {monster}**")
                    st.caption(
                        f"补强收益分: {item['total_gain_score']} | "
                        f"单卡覆盖+{item['single_gain']} | "
                        f"双卡覆盖+{item['pair_gain']} | "
                        f"阵容覆盖+{item['lineup_gain']}"
                    )
                    render_suggestion_action_buttons([monster], f"missing_add_{idx}_{monster}")
        else:
            st.caption("暂无明显的关键缺失拼图。")

    with right:
        st.markdown("#### 当前 Box 中覆盖贡献较低的魔灵")
        if analysis["low_value_owned"]:
            for item in analysis["low_value_owned"]:
                monster = item["monster"]
                image_path = resolve_image_for_streamlit(image_map.get(monster))
                row = st.columns([1, 3])
                with row[0]:
                    if image_path:
                        st.image(image_path, use_container_width=True)
                with row[1]:
                    st.write(f"**{monster}**")
                    st.caption(
                        f"覆盖贡献分: {item['total_contrib']} | "
                        f"单卡贡献={item['single_contrib']} | "
                        f"双卡贡献={item['pair_contrib']} | "
                        f"阵容贡献={item['lineup_contrib']}"
                    )
        else:
            st.caption("暂无低贡献魔灵。")

def render_best_addition_analysis_result(result, image_map):
    st.markdown("### 补 1 张 / 补 3 张 的最优覆盖提升分析")

    base = result.get("base_eval", {})
    st.caption(
        f"当前综合覆盖分: {base.get('overall_score', 0):.4f} | "
        f"单卡覆盖={base.get('single_rate', 0)*100:.1f}% | "
        f"双卡覆盖={base.get('pair_rate', 0)*100:.1f}% | "
        f"阵容覆盖={base.get('lineup_rate', 0)*100:.1f}%"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### 补 1 张最优方案")
        if result["best_one"]:
            for idx, item in enumerate(result["best_one"], start=1):
                monster = item["monsters"][0]
                image_path = resolve_image_for_streamlit(image_map.get(monster))
                row = st.columns([1, 3])
                with row[0]:
                    if image_path:
                        st.image(image_path, use_container_width=True)
                with row[1]:
                    st.write(f"**{idx}. {monster}**")
                    st.caption(
                        f"综合提升: +{item['gain']:.4f} | "
                        f"单卡: +{item['single_gain']*100:.1f}% | "
                        f"双卡: +{item['pair_gain']*100:.1f}% | "
                        f"阵容: +{item['lineup_gain']*100:.1f}%"
                    )
                    render_suggestion_action_buttons([monster], f"best_one_{idx}_{monster}")
        else:
            st.caption("暂无补 1 张建议。")

    with right:
        st.markdown("#### 补 3 张最优方案")
        if result["best_three"]:
            for idx, item in enumerate(result["best_three"], start=1):
                monsters = item["monsters"]
                img_cols = st.columns(3)
                for i, m in enumerate(monsters):
                    with img_cols[i]:
                        image_path = resolve_image_for_streamlit(image_map.get(m))
                        if image_path:
                            st.image(image_path, caption=m, use_container_width=True)
                        else:
                            st.caption(m)

                st.write(f"**{idx}. {' + '.join(monsters)}**")
                st.caption(
                    f"综合提升: +{item['gain']:.4f} | "
                    f"单卡: +{item['single_gain']*100:.1f}% | "
                    f"双卡: +{item['pair_gain']*100:.1f}% | "
                    f"阵容: +{item['lineup_gain']*100:.1f}%"
                )
                render_suggestion_action_buttons(monsters, f"best_three_{idx}_{'_'.join(monsters)}")
        else:
            st.caption("暂无补 3 张建议。")

def render_monster_card(monster, image_map, unique_key):
    image_path = resolve_image_for_streamlit(image_map.get(monster))

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-weight:700; font-size:10px; text-align:center; margin-bottom:2px; line-height:1.2;">
                {monster}
            </div>
            """,
            unsafe_allow_html=True
        )

        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.caption("无图")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("我", key=f"my_{monster}_{unique_key}"):
                add_unique("my_picks", monster, max_len=5)
        with c2:
            if st.button("敌", key=f"enemy_{monster}_{unique_key}"):
                add_unique("enemy_picks", monster, max_len=5)
        with c3:
            if st.button("禁", key=f"ban_{monster}_{unique_key}"):
                add_ban(monster)

def render_box_click_card(monster, image_map, unique_key):
    image_path = resolve_image_for_streamlit(image_map.get(monster))
    owned = monster in st.session_state.my_box_state

    border_color = "#2ecc71" if owned else "#888888"
    badge_text = "已拥有" if owned else "点击加入"
    encoded = image_file_to_base64(image_path) if image_path else None

    card_html = f"""
    <div style="
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 4px;
        text-align: center;
        min-height: 170px;
        background: rgba(255,255,255,0.02);
    ">
        <div style="font-size:10px;font-weight:700;line-height:1.2;margin-bottom:4px;">{monster}</div>
        <div style="font-size:9px;color:{border_color};margin-bottom:4px;">{badge_text}</div>
        {"<img src='data:image/png;base64," + encoded + "' style='width:100%;border-radius:6px;' />" if encoded else "<div style='height:100px;display:flex;align-items:center;justify-content:center;border:1px dashed #666;border-radius:6px;'>无图</div>"}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
    if st.button("点击图片区域切换", key=f"box_click_{monster}_{unique_key}"):
        toggle_my_box(monster)

def render_plan_card(plan_obj, image_map, family_map, idx, first_pick_side):
    plan = plan_obj["plan"]
    style_tags = get_plan_style_tags(plan, family_map)

    st.markdown(f"### 方案 {idx}")
    st.caption(" / ".join(style_tags))

    img_cols = st.columns(len(plan))
    for i, monster in enumerate(plan):
        with img_cols[i]:
            image_path = resolve_image_for_streamlit(image_map.get(monster))
            if image_path:
                st.image(image_path, caption=monster, use_container_width=True)
            else:
                st.write(monster)

    st.caption(f"综合分: {plan_obj['plan_score']}")
    st.caption(f"预估胜率: {plan_obj['estimated_win_rate']}")

    if st.button("采用方案", key=f"apply_pick_plan_{idx}"):
        apply_pick_plan(plan, first_pick_side)

def render_enemy_prediction_card(pred_obj, image_map, idx):
    monster = pred_obj["monster"]
    image_path = resolve_image_for_streamlit(image_map.get(monster))

    st.markdown(f"### 预测 {idx}")

    if image_path:
        st.image(image_path, caption=monster, use_container_width=True)
    else:
        st.write(monster)

    st.caption(f"预估分数: {pred_obj['final_score']}")
    st.caption(f"预估胜率: {pred_obj['estimated_win_rate']}")

def render_ban_card(ban_obj, image_map, idx):
    monster = ban_obj["ban_target"]
    image_path = resolve_image_for_streamlit(image_map.get(monster))

    st.markdown(f"### Ban {idx}")

    if image_path:
        st.image(image_path, caption=monster, use_container_width=True)
    else:
        st.write(monster)

    st.caption(f"Ban目标: {monster}")
    st.caption(f"预估胜率: {ban_obj['estimated_win_rate']}")
    st.caption(f"综合分: {ban_obj['final_score']}")

    if st.button("采用 Ban", key=f"apply_ban_plan_{idx}"):
        apply_ban_plan(monster, ban_obj["estimated_win_rate"])

samples, recommender, pick_freq, coverage_stats = load_data()
monster_pool = recommender.monster_pool
my_box = load_json_file(MY_BOX_PATH, [])
image_map = load_json_file(IMAGE_MAP_PATH, {})
attr_map = load_json_file(ATTR_MAP_PATH, {})
family_map = load_json_file(FAMILY_MAP_PATH, {})

attributes = ["全部"] + sorted(set(attr_map.values()))
families = ["全部"] + sorted(set(family_map.values()))

st.set_page_config(page_title="魔灵召唤 BP 推荐器", layout="wide")
init_state(initial_my_box=my_box if isinstance(my_box, list) else [])

st.title("魔灵召唤 BP 推荐器")
st.caption("提速优化版：BP 推荐优先优化 + Box 分析按需生成")

st.markdown("""
<style>
div.stButton > button {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    padding: 0.12rem 0.05rem !important;
    min-height: 28px !important;
    font-size: 10px !important;
    line-height: 1 !important;
}
div.stButton > button p,
div.stButton > button span {
    width: 100% !important;
    text-align: center !important;
    justify-content: center !important;
    margin: 0 !important;
    font-size: 10px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("对局参数")

    first_pick_side = st.selectbox(
        "谁是一选",
        options=["me", "enemy"],
        format_func=lambda x: "我方一选" if x == "me" else "对方一选"
    )

    top_k = st.slider("推荐方案数", 1, 10, 5)
    min_similarity = st.slider("最小相似度", 0.0, 20.0, 3.0, 0.5)
    min_support = st.slider("最小支持样本数", 1, 20, 1)
    candidate_pool_size = st.slider("双选组合候选池大小", 4, 20, 12)

    st.divider()
    st.header("BP卡片筛选")
    only_owned_in_picker = st.checkbox("卡片区只显示我拥有的魔灵", value=False)
    use_my_box_for_reco = st.checkbox("推荐时只推荐我拥有的魔灵", value=False)
    only_with_image = st.checkbox("只显示有图片的魔灵", value=False)

    attr_filter = st.selectbox("属性筛选", attributes)
    family_filter = st.selectbox("系列筛选", families)

    st.divider()
    st.header("显示性能")
    bp_max_display_count = st.slider("BP图库最多显示数量", 20, 300, 80, 10)
    box_max_display_count = st.slider("Box图库最多显示数量", 20, 300, 80, 10)

update_phase_hint(first_pick_side)

tab_bp, tab_box = st.tabs(["BP 模式", "我的 Box 管理"])

with tab_bp:
    left, right = st.columns([4.5, 2])

    with left:
        st.subheader("魔灵图库")

        search_text = st.text_input("搜索魔灵", value=st.session_state.search_text)
        st.session_state.search_text = search_text

        filtered_monsters = filter_monsters(
            monster_pool=monster_pool,
            search_text=search_text,
            attr_filter=attr_filter,
            family_filter=family_filter,
            only_owned=only_owned_in_picker,
            my_box=st.session_state.my_box_state if isinstance(st.session_state.my_box_state, list) else [],
            attr_map=attr_map,
            family_map=family_map,
            only_with_image=only_with_image,
            image_map=image_map
        )

        filtered_monsters = sort_monsters_by_pick_frequency(filtered_monsters, pick_freq)
        filtered_monsters = filtered_monsters[:bp_max_display_count]

        st.caption(f"共显示 {len(filtered_monsters)} 个魔灵")

        columns_per_row = 10
        rows = math.ceil(len(filtered_monsters) / columns_per_row)

        for row in range(rows):
            cols = st.columns(columns_per_row)
            row_items = filtered_monsters[row * columns_per_row:(row + 1) * columns_per_row]

            for col_idx, (col, monster) in enumerate(zip(cols, row_items)):
                with col:
                    render_monster_card(monster, image_map, f"bp_{row}_{col_idx}")

    with right:
        st.subheader("当前 BP 状态")

        st.info(f"阶段提示：{st.session_state.phase_hint}")

        if st.session_state.last_action:
            st.success(st.session_state.last_action)

        render_selected_monsters("我方已选", st.session_state.my_picks, image_map, "rm_my")
        render_selected_monsters("对方已选", st.session_state.enemy_picks, image_map, "rm_enemy")
        render_selected_monsters("不可选 / 预禁", st.session_state.banned_monsters, image_map, "rm_ban")

        st.markdown("### 最终 Ban")
        if st.session_state.final_bans:
            render_selected_monsters("已采用的最终 Ban", st.session_state.final_bans, image_map, "rm_final_ban")
        else:
            st.caption("尚未选择最终 Ban")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("清空全部"):
                clear_all()

        turn = infer_turn_from_state(
            st.session_state.my_picks,
            st.session_state.enemy_picks,
            first_pick_side
        )

        st.markdown("### 当前状态信息")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("我方已选", len(st.session_state.my_picks))
        cc2.metric("对方已选", len(st.session_state.enemy_picks))
        cc3.metric("当前手数", turn if turn is not None else "非我方回合")

        owned_monsters = st.session_state.my_box_state if (
            use_my_box_for_reco and isinstance(st.session_state.my_box_state, list) and st.session_state.my_box_state
        ) else None

        if turn is not None:
            if st.button("生成 Pick 推荐", type="primary"):
                st.session_state.pick_plans = recommender.recommend_pick_plans(
                    my_picks=st.session_state.my_picks,
                    enemy_picks=st.session_state.enemy_picks,
                    first_pick_side=first_pick_side,
                    turn=turn,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    min_support=min_support,
                    banned_monsters=st.session_state.banned_monsters,
                    owned_monsters=owned_monsters,
                    candidate_pool_size=candidate_pool_size
                )
                st.session_state.ban_plans = []
                st.session_state.enemy_predictions = []

            if st.session_state.pick_plans:
                st.markdown("### Pick 推荐方案")
                cols = st.columns(len(st.session_state.pick_plans))
                for idx, (col, plan_obj) in enumerate(zip(cols, st.session_state.pick_plans), start=1):
                    with col:
                        with st.container(border=True):
                            render_plan_card(plan_obj, image_map, family_map, idx, first_pick_side)

        else:
            st.warning("当前不是我方选人回合，通常表示现在轮到对方操作。")

            if len(st.session_state.my_picks) < 5 and len(st.session_state.enemy_picks) < 5:
                st.markdown("### 对方下一手常见 Pick")

                if st.button("生成对方预测"):
                    st.session_state.enemy_predictions = recommender.recommend_enemy_next_picks(
                        my_picks=st.session_state.my_picks,
                        enemy_picks=st.session_state.enemy_picks,
                        first_pick_side=first_pick_side,
                        top_k=5,
                        min_similarity=min_similarity,
                        min_support=min_support,
                        banned_monsters=st.session_state.banned_monsters,
                        owned_monsters=None
                    )
                    st.session_state.pick_plans = []

                if st.session_state.enemy_predictions:
                    cols = st.columns(len(st.session_state.enemy_predictions))
                    for idx, (col, pred_obj) in enumerate(zip(cols, st.session_state.enemy_predictions), start=1):
                        with col:
                            with st.container(border=True):
                                render_enemy_prediction_card(pred_obj, image_map, idx)

        if len(st.session_state.my_picks) == 5 and len(st.session_state.enemy_picks) == 5:
            st.markdown("### Ban 胜率预测")

            if st.button("生成 Ban 推荐", type="secondary"):
                st.session_state.ban_plans = recommender.recommend_bans(
                    my_picks=st.session_state.my_picks,
                    enemy_picks=st.session_state.enemy_picks,
                    top_k=5,
                    min_similarity=5.0,
                    min_support=1
                )
                st.session_state.enemy_predictions = []

            if st.session_state.ban_plans:
                cols = st.columns(len(st.session_state.ban_plans))
                for idx, (col, ban_obj) in enumerate(zip(cols, st.session_state.ban_plans), start=1):
                    with col:
                        with st.container(border=True):
                            render_ban_card(ban_obj, image_map, idx)

with tab_box:
    st.subheader("我的 Box 管理")
    st.caption("提速优化：分析改为按按钮生成，图片显示数量可限制。")

    render_current_box_top(st.session_state.my_box_state, image_map, pick_freq)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        box_search = st.text_input("搜索我的 Box 魔灵", key="box_search_text")
    with col_b:
        box_attr_filter = st.selectbox("属性筛选", attributes, key="box_attr_filter")
    with col_c:
        box_family_filter = st.selectbox("系列筛选", families, key="box_family_filter")

    box_only_owned = st.checkbox("只显示已拥有魔灵", value=False, key="box_only_owned")
    box_only_with_image = st.checkbox("只显示有图片的魔灵", value=True, key="box_only_with_image")

    box_filtered_monsters = filter_monsters(
        monster_pool=monster_pool,
        search_text=box_search,
        attr_filter=box_attr_filter,
        family_filter=box_family_filter,
        only_owned=False,
        my_box=st.session_state.my_box_state,
        attr_map=attr_map,
        family_map=family_map,
        only_with_image=box_only_with_image,
        image_map=image_map
    )

    if box_only_owned:
        box_filtered_monsters = [m for m in box_filtered_monsters if m in st.session_state.my_box_state]

    box_filtered_monsters = sort_monsters_by_pick_frequency(box_filtered_monsters, pick_freq)
    box_filtered_monsters = box_filtered_monsters[:box_max_display_count]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("保存我的 Box", type="primary"):
            save_my_box()
    with c2:
        if st.button("重新载入 my_box.json"):
            latest_box = load_json_file(MY_BOX_PATH, [])
            st.session_state.my_box_state = latest_box if isinstance(latest_box, list) else []
            st.session_state.last_action = f"已重新载入 my_box.json，共 {len(st.session_state.my_box_state)} 个魔灵"
            invalidate_box_analysis()
    with c3:
        if st.button("清空我的 Box"):
            clear_my_box()
    with c4:
        if st.button("批量全选当前筛选结果"):
            bulk_add_to_box(box_filtered_monsters)
    with c5:
        if st.button("批量取消当前筛选结果"):
            bulk_remove_from_box(box_filtered_monsters)

    st.info(f"当前我的 Box 数量：{len(st.session_state.my_box_state)}")
    st.caption(f"当前筛选结果数量：{len(box_filtered_monsters)}")

    st.markdown("### Box 分析")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("生成 Box 自洽分析"):
            st.session_state.box_analysis_result = analyze_box(
                st.session_state.my_box_state,
                monster_pool,
                attr_map,
                family_map,
                samples
            )
    with a2:
        if st.button("生成覆盖分析"):
            st.session_state.coverage_analysis_result = build_box_coverage_analysis(
                st.session_state.my_box_state,
                coverage_stats
            )
    with a3:
        if st.button("生成补强分析"):
            st.session_state.best_addition_result = find_best_additions(
                st.session_state.my_box_state,
                coverage_stats
            )

    if st.session_state.box_analysis_result is not None:
        st.markdown("### Box 自洽分析")
        box_analysis = st.session_state.box_analysis_result
        score = box_analysis["score"]
        st.metric("Box 自洽评分", score)

        for line in box_analysis["summary"]:
            st.caption(line)

        col_add, col_remove = st.columns(2)
        with col_add:
            st.markdown("#### 建议新增")
            if box_analysis["add_suggestions"]:
                for idx, item in enumerate(box_analysis["add_suggestions"], start=1):
                    monster = item["monster"]
                    st.write(f"**{monster}**")
                    st.caption(item["reason"])
                    render_suggestion_action_buttons([monster], f"box_add_suggestion_{idx}_{monster}")
            else:
                st.caption("暂无明显新增建议。")

        with col_remove:
            st.markdown("#### 建议删除")
            if box_analysis["remove_suggestions"]:
                for item in box_analysis["remove_suggestions"]:
                    st.write(f"**{item['monster']}**")
                    st.caption(item["reason"])
            else:
                st.caption("暂无明显删除建议。")

    if st.session_state.coverage_analysis_result is not None:
        render_box_coverage_analysis_result(
            st.session_state.coverage_analysis_result,
            image_map
        )

    if st.session_state.best_addition_result is not None:
        render_best_addition_analysis_result(
            st.session_state.best_addition_result,
            image_map
        )

    st.markdown("### 点击图片区域管理 Box")
    st.caption(f"当前显示 {len(box_filtered_monsters)} 个魔灵")

    columns_per_row = 10
    rows = math.ceil(len(box_filtered_monsters) / columns_per_row)

    for row in range(rows):
        cols = st.columns(columns_per_row)
        row_items = box_filtered_monsters[row * columns_per_row:(row + 1) * columns_per_row]

        for col_idx, (col, monster) in enumerate(zip(cols, row_items)):
            with col:
                render_box_click_card(monster, image_map, f"box_{row}_{col_idx}")

st.divider()
with st.expander("调试信息"):
    st.write({
        "sample_count": len(samples),
        "monster_pool_size": len(monster_pool),
        "image_map_size": len(image_map),
        "attr_map_size": len(attr_map),
        "family_map_size": len(family_map),
        "my_box_size_file": len(my_box) if isinstance(my_box, list) else 0,
        "my_box_size_state": len(st.session_state.my_box_state) if isinstance(st.session_state.my_box_state, list) else 0,
        "my_picks": st.session_state.my_picks,
        "enemy_picks": st.session_state.enemy_picks,
        "banned_monsters": st.session_state.banned_monsters,
        "final_bans": st.session_state.final_bans,
        "phase_hint": st.session_state.phase_hint
    })