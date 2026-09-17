from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
import sqlite3
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from bp_ban_value_recommender import BPBanValueRecommender
from bp_lucksack_stats import BPLucksackStats
from bp_relation_common import infer_pick_count
from bp_relation_recommender import BPRelationRecommender
from bp_transformer_common import normalize_name, slugify_fallback


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
JSON_DIR = ROOT_DIR / "json"
DATA_DIR = ROOT_DIR / "data"
BP_PICKER_COMPONENT_DIR = ROOT_DIR / "components" / "bp_icon_picker"
BP_ICON_PICKER = components.declare_component(
    "bp_icon_picker",
    path=str(BP_PICKER_COMPONENT_DIR),
)

MY_BOX_PATH = JSON_DIR / "my_box.json"
IMAGE_MAP_PATH = JSON_DIR / "monster_images.json"
ATTR_MAP_PATH = JSON_DIR / "monster_attributes.json"
FAMILY_MAP_PATH = JSON_DIR / "monster_families.json"
SAMPLES_DB_PATH = DATA_DIR / "bp_samples.sqlite"
LUCK_STATS_PATH = JSON_DIR / "lucksack_stats.json"
LUCK_CATALOG_PATH = JSON_DIR / "lucksack_monster_catalog.json"
SWKIT_ALIAS_PATH = JSON_DIR / "swkit_monster_aliases.json"

RELATION_BUNDLES = (
    (
        "relation v2",
        MODEL_DIR / "bp_relation_net_v2.pt",
        MODEL_DIR / "bp_relation_meta_v2.json",
        MODEL_DIR / "bp_relation_vocab_v2.json",
    ),
    (
        "relation legacy",
        MODEL_DIR / "bp_relation_net.pt",
        MODEL_DIR / "bp_relation_meta.json",
        MODEL_DIR / "bp_relation_vocab.json",
    ),
)
PREFIX_VALUE_BUNDLE = (
    MODEL_DIR / "bp_prefix_value_net.pt",
    MODEL_DIR / "bp_prefix_value_meta.json",
    MODEL_DIR / "bp_prefix_value_vocab.json",
)
BAN_VALUE_BUNDLE = (
    MODEL_DIR / "bp_draft_value_net.pt",
    MODEL_DIR / "bp_draft_value_meta.json",
    MODEL_DIR / "bp_draft_value_vocab.json",
)

TURN_PATTERN = ("me", "enemy", "enemy", "me", "me", "enemy", "enemy", "me", "me", "enemy")
BP_GALLERY_COLUMNS = 17
BP_GALLERY_PAGE_ROWS = 3
BP_GALLERY_PAGE_SIZE = BP_GALLERY_COLUMNS * BP_GALLERY_PAGE_ROWS


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def bundle_ready(bundle):
    return all(Path(path).exists() for path in bundle)


def select_relation_bundle():
    for label, model_path, meta_path, vocab_path in RELATION_BUNDLES:
        if bundle_ready((model_path, meta_path, vocab_path)):
            return {
                "label": label,
                "model_path": model_path,
                "meta_path": meta_path,
                "vocab_path": vocab_path,
            }
    return None


def normalize_key(value):
    return normalize_name(value).replace("_", " ").replace("-", " ")


def value_variants(value):
    text = str(value or "").strip()
    if not text:
        return set()
    return {
        normalize_key(text),
        normalize_name(text),
        normalize_key(text.replace("-", " ")),
        normalize_key(text.replace(" ", "-")),
        normalize_key(slugify_fallback(text)),
    }


def resolve_image_path(value):
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        return raw

    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                ROOT_DIR / path,
                ROOT_DIR / "images" / path.name,
                ROOT_DIR / "images" / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_image_lookup(image_map):
    lookup = {}
    for key, value in (image_map or {}).items():
        image_path = resolve_image_path(value)
        if not image_path:
            continue
        for variant in value_variants(key):
            lookup[variant] = image_path
    return lookup


def get_image_path(name, image_lookup):
    for variant in value_variants(name):
        image_path = image_lookup.get(variant)
        if image_path:
            return image_path
    return None


@st.cache_data(show_spinner=False)
def load_image_data_uri(path):
    if str(path).startswith(("http://", "https://")):
        return str(path)
    image_path = Path(path)
    if not image_path.exists():
        return ""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_monster_options(vocab, display_name_map, frequency_map, fallback_names):
    options = []
    seen = set()
    for slug, raw_id in sorted((vocab or {}).items(), key=lambda item: int(item[1])):
        try:
            if int(raw_id) <= 1:
                continue
        except (TypeError, ValueError):
            continue
        display_name = str(display_name_map.get(slug) or slug).strip()
        if not display_name:
            continue
        key = normalize_name(display_name)
        if key in seen:
            continue
        seen.add(key)
        raw_frequency = frequency_map.get(slug, frequency_map.get(str(raw_id), 0))
        try:
            frequency = int(float(raw_frequency or 0))
        except (TypeError, ValueError):
            frequency = 0
        options.append((display_name, frequency))

    options.extend((str(name).strip(), 0) for name in fallback_names if str(name).strip())

    unique = {}
    for name, frequency in options:
        unique.setdefault(normalize_name(name), (name, frequency))
    return [
        name
        for name, _frequency in sorted(
            unique.values(),
            key=lambda item: (-int(item[1]), normalize_name(item[0])),
        )
    ]


@st.cache_data(show_spinner=False)
def load_web_assets():
    relation_bundle = select_relation_bundle()
    relation_meta = read_json(relation_bundle["meta_path"], {}) if relation_bundle else {}
    relation_vocab = read_json(relation_bundle["vocab_path"], {}) if relation_bundle else {}
    image_map = read_json(IMAGE_MAP_PATH, {})
    attr_map = read_json(ATTR_MAP_PATH, {})
    family_map = read_json(FAMILY_MAP_PATH, {})
    luck_stats = BPLucksackStats(stats_path=LUCK_STATS_PATH, catalog_path=LUCK_CATALOG_PATH)
    swkit_alias_data = read_json(SWKIT_ALIAS_PATH, {})
    my_box = read_json(MY_BOX_PATH, [])
    if not isinstance(my_box, list):
        my_box = []

    remote_catalog = luck_stats.catalog if isinstance(luck_stats.catalog, list) else []
    remote_image_map = {}
    remote_names = set()
    alias_map = {}
    catalog_slugs = set()
    for item in remote_catalog:
        if not item.get("searchable", True) or int(item.get("natural_stars") or 0) < 4:
            continue
        catalog_slugs.add(normalize_name(item.get("slug")))
        display_name = luck_stats.to_display_name(item.get("slug"))
        if not display_name:
            continue
        remote_names.add(display_name)
        unit_id = str(item.get("com2us_id") or "")
        aliases = (swkit_alias_data.get("aliases_by_id") or {}).get(unit_id, [])
        if aliases:
            alias_map[display_name] = sorted(set(str(alias) for alias in aliases if str(alias).strip()))
        image_name = str(item.get("image") or "").strip()
        if image_name:
            remote_image_map[display_name] = f"https://static.lucksack.gg/images/monsters/{image_name}"
        attr_map.setdefault(display_name, str(item.get("element") or ""))

    for slug in luck_stats.candidate_slugs():
        if slug in catalog_slugs:
            continue
        display_name = luck_stats.to_display_name(slug)
        if not display_name:
            continue
        remote_names.add(display_name)
        image_name = luck_stats.to_image_name(slug)
        if image_name:
            remote_image_map[display_name] = (
                f"https://static.lucksack.gg/images/monsters/{image_name}"
            )

    fallback_names = (
        set(image_map)
        | set(attr_map)
        | set(family_map)
        | set(my_box)
        | remote_names
    )
    options = build_monster_options(
        relation_vocab,
        relation_meta.get("display_name_map", {}) or {},
        relation_meta.get("monster_frequency", {}) or {},
        fallback_names,
    )
    option_set = set(options)
    my_box = [name for name in my_box if name in option_set]
    search_text_map = {
        name: " ".join(
            [name, slugify_fallback(name)]
            + list(alias_map.get(name, []))
        ).lower()
        for name in options
    }

    return {
        "relation_bundle": relation_bundle,
        "prefix_value_ready": bundle_ready(PREFIX_VALUE_BUNDLE),
        "ban_value_ready": bundle_ready(BAN_VALUE_BUNDLE),
        "options": options,
        "my_box": sorted(set(my_box), key=normalize_name),
        "image_lookup": {
            **build_image_lookup(image_map),
            **build_image_lookup(remote_image_map),
        },
        "attr_map": attr_map,
        "family_map": family_map,
        "alias_map": alias_map,
        "search_text_map": search_text_map,
        "attributes": ["全部"] + sorted({str(value) for value in attr_map.values() if str(value).strip()}),
        "families": ["全部"] + sorted({str(value) for value in family_map.values() if str(value).strip()}),
        "luck_stats_ready": bool(luck_stats.ready),
        "luck_stats_source": luck_stats.source_label,
    }


@st.cache_data(show_spinner=False)
def load_sample_count(path):
    if not Path(path).exists():
        return 0
    try:
        with sqlite3.connect(str(path)) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM bp_samples").fetchone()[0])
    except sqlite3.Error:
        return 0


@st.cache_resource(show_spinner=False)
def load_relation_recommender(model_path, meta_path, vocab_path, device):
    selected_device = None if device == "auto" else device
    return BPRelationRecommender(
        model_path=model_path,
        meta_path=meta_path,
        vocab_path=vocab_path,
        device=selected_device,
    )


@st.cache_resource(show_spinner=False)
def load_prefix_value_recommender(model_path, meta_path, vocab_path, device):
    selected_device = None if device == "auto" else device
    from bp_prefix_value_recommender import BPPrefixValueRecommender

    return BPPrefixValueRecommender(
        model_path=model_path,
        meta_path=meta_path,
        vocab_path=vocab_path,
        device=selected_device,
    )


@st.cache_resource(show_spinner=False)
def load_ban_value_recommender(model_path, meta_path, vocab_path, device):
    selected_device = None if device == "auto" else device
    return BPBanValueRecommender(
        model_path=model_path,
        meta_path=meta_path,
        vocab_path=vocab_path,
        device=selected_device,
    )


def side_for_turn(turn, first_pick_side):
    if turn < 1 or turn > len(TURN_PATTERN):
        return None
    first_pick_side = normalize_name(first_pick_side)
    if first_pick_side not in {"me", "enemy"}:
        return None
    if first_pick_side == "enemy":
        swapped = tuple("enemy" if side == "me" else "me" for side in TURN_PATTERN)
        return swapped[turn - 1]
    return TURN_PATTERN[turn - 1]


def phase_label(my_picks, enemy_picks, current_side):
    if len(my_picks) >= 5 and len(enemy_picks) >= 5:
        return "Ban 阶段"
    if current_side == "me":
        return "我方选人"
    if current_side == "enemy":
        return "对方选人"
    return "等待输入"


def make_signature(my_picks, enemy_picks, first_pick_side, my_leader, enemy_leader):
    payload = {
        "my_picks": list(my_picks),
        "enemy_picks": list(enemy_picks),
        "first_pick_side": first_pick_side,
        "my_leader": my_leader,
        "enemy_leader": enemy_leader,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def current_device_label(device):
    if device == "auto":
        return "自动"
    return device


def initialize_state(assets):
    defaults = {
        "first_pick_side": "me",
        "my_picks": [],
        "enemy_picks": [],
        "my_leader": "",
        "enemy_leader": "",
        "my_box": list(assets["my_box"]),
        "use_my_box": True,
        "top_k": 6,
        "max_pair_candidates": 400,
        "show_details": False,
        "device": "auto",
        "gallery_search": "",
        "gallery_attr": "全部",
        "gallery_family": "全部",
        "gallery_only_owned": False,
        "gallery_only_with_image": True,
        "gallery_page": 1,
        "picker_event_id": "",
        "box_search": "",
        "box_page": 1,
        "pick_results": None,
        "enemy_results": None,
        "ban_results": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    options = set(assets["options"])
    st.session_state.my_picks = [item for item in st.session_state.my_picks if item in options]
    st.session_state.enemy_picks = [item for item in st.session_state.enemy_picks if item in options]
    st.session_state.my_box = [item for item in st.session_state.my_box if item in options]
    if st.session_state.my_leader not in st.session_state.my_picks:
        st.session_state.my_leader = ""
    if st.session_state.enemy_leader not in st.session_state.enemy_picks:
        st.session_state.enemy_leader = ""


def clear_results():
    st.session_state.pick_results = None
    st.session_state.enemy_results = None
    st.session_state.ban_results = None


def apply_recommendation_plan(result_type, monsters, options):
    target = st.session_state.my_picks if result_type == "pick" else st.session_state.enemy_picks
    occupied = set(st.session_state.my_picks) | set(st.session_state.enemy_picks)
    for monster in monsters:
        if monster not in options or len(target) >= 5 or monster in occupied:
            continue
        target.append(monster)
        occupied.add(monster)
    clear_results()


def save_my_box():
    MY_BOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MY_BOX_PATH, "w", encoding="utf-8") as file:
        json.dump(sorted(set(st.session_state.my_box), key=normalize_name), file, ensure_ascii=False, indent=2)
    load_web_assets.clear()


def get_relation_model(assets):
    bundle = assets["relation_bundle"]
    if not bundle:
        raise FileNotFoundError("没有找到 Relation 模型三件套")
    return load_relation_recommender(
        str(bundle["model_path"]),
        str(bundle["meta_path"]),
        str(bundle["vocab_path"]),
        st.session_state.device,
    )


def get_prefix_model(assets):
    if not assets["prefix_value_ready"]:
        return None
    return load_prefix_value_recommender(
        *(str(path) for path in PREFIX_VALUE_BUNDLE),
        st.session_state.device,
    )


def get_ban_model(assets):
    if not assets["ban_value_ready"]:
        return None
    return load_ban_value_recommender(
        *(str(path) for path in BAN_VALUE_BUNDLE),
        st.session_state.device,
    )


LOOKAHEAD_PLAN_LIMIT = 12
LOOKAHEAD_RESPONSE_LIMIT = 3
LOOKAHEAD_PAIR_CANDIDATE_LIMIT = 80


def _plan_monsters(item):
    monsters = [value for value in item.get("plan") or item.get("monsters") or [] if value]
    if not monsters:
        monster = item.get("monster") or item.get("display_name")
        monsters = [monster] if monster else []
    return monsters


def _prefix_state_value(prefix_model, state, my_picks, enemy_picks, turn, current_side, cache):
    key = (
        tuple(my_picks),
        tuple(enemy_picks),
        int(turn),
        normalize_name(state["first_pick_side"]),
        normalize_name(current_side),
        state.get("my_leader", ""),
        state.get("enemy_leader", ""),
    )
    if key not in cache:
        evaluation = prefix_model.evaluate_state(
            my_picks=list(my_picks),
            enemy_picks=list(enemy_picks),
            turn=turn,
            first_pick_side=state["first_pick_side"],
            current_side=current_side,
            my_leader=state["my_leader"],
            enemy_leader=state["enemy_leader"],
            return_detail=True,
        )
        cache[key] = float(evaluation.get("calibrated_win_rate", 0.5))
    return cache[key]


def _evidence_score(detail):
    values = []
    if (
        float(detail.get("draft_stats_coverage", 0.0) or 0.0) > 0.0
        or detail.get("draft_stats_source") not in {None, "unavailable"}
    ):
        values.append(float(detail.get("draft_stats_score", 0.5) or 0.5))
    if bool(detail.get("mechanism_available")):
        values.append(float(detail.get("mechanism_score", 0.5) or 0.5))
    if not values:
        return 0.5
    return sum(values) / len(values)


def _weighted_response_value(response_rows):
    if not response_rows:
        return None, None, []
    values = [float(row["value"]) for row in response_rows]
    priorities = [float(row.get("priority", 0.5)) for row in response_rows]
    highest_priority = max(priorities)
    weights = [math.exp((priority - highest_priority) / 0.05) for priority in priorities]
    weight_sum = sum(weights) or 1.0
    expected = sum(value * weight for value, weight in zip(values, weights)) / weight_sum
    return min(values), expected, response_rows


def _my_action_lookahead(plan, relation_model, prefix_model, state, assets, cache):
    picked = _plan_monsters(plan)
    if not picked:
        return None
    post_my = list(state["my_picks"]) + picked
    post_turn = int(state["turn"]) + len(picked)
    post_side = side_for_turn(post_turn, state["first_pick_side"]) or "enemy"
    responses = []
    if post_side == "enemy":
        counter_plans = relation_model.recommend_enemy_next_picks(
            my_picks=post_my,
            enemy_picks=state["enemy_picks"],
            first_pick_side=state["first_pick_side"],
            top_k=LOOKAHEAD_RESPONSE_LIMIT,
            candidate_pool=assets["options"],
            max_pair_candidates=min(
                int(state.get("max_pair_candidates", LOOKAHEAD_PAIR_CANDIDATE_LIMIT)),
                LOOKAHEAD_PAIR_CANDIDATE_LIMIT,
            ),
        )
        for counter_plan in counter_plans:
            counter = _plan_monsters(counter_plan)
            if not counter:
                continue
            next_turn = post_turn + len(counter)
            next_side = side_for_turn(next_turn, state["first_pick_side"]) or "me"
            responses.append(
                {
                    "plan": counter,
                    "priority": float(counter_plan.get("final_score", 0.5)),
                    "value": _prefix_state_value(
                        prefix_model,
                        state,
                        post_my,
                        list(state["enemy_picks"]) + counter,
                        next_turn,
                        next_side,
                        cache,
                    ),
                }
            )
    worst, expected, rows = _weighted_response_value(responses)
    immediate = _prefix_state_value(
        prefix_model,
        state,
        post_my,
        state["enemy_picks"],
        post_turn,
        post_side,
        cache,
    )
    robust = immediate if worst is None else 0.65 * worst + 0.35 * expected
    return {
        "immediate": immediate,
        "robust": robust,
        "worst_counter": worst,
        "mean_counter": expected,
        "responses": rows,
    }


def _enemy_action_lookahead(plan, relation_model, prefix_model, state, assets, cache):
    picked = _plan_monsters(plan)
    if not picked:
        return None
    post_enemy = list(state["enemy_picks"]) + picked
    post_turn = int(state["turn"]) + len(picked)
    post_side = side_for_turn(post_turn, state["first_pick_side"]) or "me"
    immediate = _prefix_state_value(
        prefix_model,
        state,
        state["my_picks"],
        post_enemy,
        post_turn,
        post_side,
        cache,
    )
    responses = []
    if post_side == "me":
        response_plans = relation_model.recommend_pick_plans(
            my_picks=state["my_picks"],
            enemy_picks=post_enemy,
            first_pick_side=state["first_pick_side"],
            candidate_pool=assets["options"],
            my_box=state.get("my_box"),
            use_my_box_for_reco=bool(state.get("use_my_box", True)),
            top_k=LOOKAHEAD_RESPONSE_LIMIT,
            max_pair_candidates=min(
                int(state.get("max_pair_candidates", LOOKAHEAD_PAIR_CANDIDATE_LIMIT)),
                LOOKAHEAD_PAIR_CANDIDATE_LIMIT,
            ),
        )
        for response_plan in response_plans:
            response = _plan_monsters(response_plan)
            if not response:
                continue
            next_turn = post_turn + len(response)
            next_side = side_for_turn(next_turn, state["first_pick_side"]) or "enemy"
            responses.append(
                {
                    "plan": response,
                    "priority": float(response_plan.get("plan_score", 0.5)),
                    "value": _prefix_state_value(
                        prefix_model,
                        state,
                        list(state["my_picks"]) + response,
                        post_enemy,
                        next_turn,
                        next_side,
                        cache,
                    ),
                }
            )
    best_response = max((row["value"] for row in responses), default=immediate)
    return {
        "immediate": immediate,
        "best_response": best_response,
        "responses": responses,
    }


def rerank_my_plans(plans, prefix_model, relation_model, state, assets, top_k):
    if not plans or prefix_model is None:
        return plans[:top_k]

    value_cache = {}
    base_win_rate = _prefix_state_value(
        prefix_model,
        state,
        state["my_picks"],
        state["enemy_picks"],
        state["turn"],
        side_for_turn(state["turn"], state["first_pick_side"]) or "me",
        value_cache,
    )
    enriched = []
    for plan_index, plan in enumerate(plans):
        item = dict(plan)
        picked = _plan_monsters(item)
        detail = dict(item.get("detail", {}) or {})
        try:
            lookahead = None
            if plan_index < LOOKAHEAD_PLAN_LIMIT and relation_model is not None:
                lookahead = _my_action_lookahead(
                    item,
                    relation_model,
                    prefix_model,
                    state,
                    assets,
                    value_cache,
                )
            if lookahead is None:
                post_turn = state["turn"] + len(picked)
                post_side = side_for_turn(post_turn, state["first_pick_side"]) or "enemy"
                win_rate = _prefix_state_value(
                    prefix_model,
                    state,
                    state["my_picks"] + picked,
                    state["enemy_picks"],
                    post_turn,
                    post_side,
                    value_cache,
                )
                robust_win_rate = win_rate
            else:
                win_rate = float(lookahead["immediate"])
                robust_win_rate = float(lookahead["robust"])
            relation_score = float(
                detail.get("normalized_relation_score", item.get("plan_score", 0.0)) or 0.0
            )
            evidence_score = _evidence_score(detail)
            fused_score = (
                0.60 * robust_win_rate
                + 0.15 * win_rate
                + 0.15 * relation_score
                + 0.10 * evidence_score
            )
            item["plan_score"] = round(fused_score, 6)
            item["estimated_win_rate"] = round(win_rate, 6)
            detail.update(
                {
                    "pre_pick_win_rate": round(base_win_rate, 6),
                    "post_pick_win_rate": round(win_rate, 6),
                    "adversarial_win_rate": round(robust_win_rate, 6),
                    "win_rate_gain": round(win_rate - base_win_rate, 6),
                    "win_rate_source": "prefix_value",
                    "relation_component": round(relation_score, 6),
                    "value_component": round(win_rate, 6),
                    "adversarial_value_component": round(robust_win_rate, 6),
                    "evidence_component": round(evidence_score, 6),
                    "lookahead_enabled": bool(lookahead),
                    "worst_counter_win_rate": (
                        None if not lookahead else lookahead.get("worst_counter")
                    ),
                    "mean_counter_win_rate": (
                        None if not lookahead else lookahead.get("mean_counter")
                    ),
                    "counter_plans": [] if not lookahead else [
                        {
                            "plan": row["plan"],
                            "win_rate": round(float(row["value"]), 6),
                            "priority": round(float(row["priority"]), 6),
                        }
                        for row in lookahead.get("responses", [])
                    ],
                    "ranking_source": "minimax_value_0.60_immediate_0.15_relation_0.15_evidence_0.10",
                }
            )
        except Exception as exc:
            detail["win_rate_source"] = "prefix_value_failed"
            detail["value_error"] = str(exc)
        item["detail"] = detail
        enriched.append(item)
    enriched.sort(key=lambda row: float(row.get("plan_score", 0.0)), reverse=True)
    return enriched[:top_k]


def rerank_enemy_plans(plans, prefix_model, relation_model, state, assets, top_k):
    if not plans or prefix_model is None:
        return plans[:top_k]

    value_cache = {}
    enriched = []
    for plan_index, plan in enumerate(plans):
        item = dict(plan)
        monsters = _plan_monsters(item)
        detail = dict(item.get("detail", {}) or {})
        try:
            next_side = side_for_turn(state["turn"] + len(monsters), state["first_pick_side"]) or "me"
            lookahead = None
            if plan_index < LOOKAHEAD_PLAN_LIMIT and relation_model is not None:
                lookahead = _enemy_action_lookahead(
                    item,
                    relation_model,
                    prefix_model,
                    state,
                    assets,
                    value_cache,
                )
            if lookahead is None:
                win_rate = _prefix_state_value(
                    prefix_model,
                    state,
                    state["my_picks"],
                    state["enemy_picks"] + monsters,
                    state["turn"] + len(monsters),
                    next_side,
                    value_cache,
                )
                best_response = win_rate
            else:
                win_rate = float(lookahead["immediate"])
                best_response = float(lookahead["best_response"])
            relation_score = float(
                detail.get("normalized_relation_score", item.get("final_score", 0.0)) or 0.0
            )
            evidence_score = _evidence_score(detail)
            adversarial_score = (
                0.60 * (1.0 - best_response)
                + 0.15 * (1.0 - win_rate)
                + 0.15 * relation_score
                + 0.10 * evidence_score
            )
            item["estimated_win_rate"] = round(win_rate, 6)
            item["final_score"] = round(adversarial_score, 6)
            detail.update(
                {
                    "post_enemy_pick_win_rate": round(win_rate, 6),
                    "best_response_win_rate": round(best_response, 6),
                    "enemy_adversarial_score": round(adversarial_score, 6),
                    "relation_component": round(relation_score, 6),
                    "evidence_component": round(evidence_score, 6),
                    "win_rate_source": "prefix_value",
                    "lookahead_enabled": bool(lookahead),
                    "response_plans": [] if not lookahead else [
                        {
                            "plan": row["plan"],
                            "win_rate": round(float(row["value"]), 6),
                            "priority": round(float(row["priority"]), 6),
                        }
                        for row in lookahead.get("responses", [])
                    ],
                    "ranking_source": "minimax_enemy_0.60_best_response_0.15_immediate_0.15_relation_0.10_evidence",
                }
            )
        except Exception as exc:
            detail["win_rate_source"] = "prefix_value_failed"
            detail["value_error"] = str(exc)
        item["detail"] = detail
        enriched.append(item)
    enriched.sort(key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
    return enriched[:top_k]


def calculate_my_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    turn = len(my_picks) + len(enemy_picks) + 1
    first_pick_side = st.session_state.first_pick_side
    pick_count = infer_pick_count(my_picks, enemy_picks, first_pick_side)
    relation_model = get_relation_model(assets)
    prefix_model = get_prefix_model(assets)
    top_k = int(st.session_state.top_k)
    plans = relation_model.recommend_pick_plans(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        pick_count=pick_count,
        turn=turn,
        first_pick_side=first_pick_side,
        candidate_pool=assets["options"],
        my_box=st.session_state.my_box,
        use_my_box_for_reco=bool(st.session_state.use_my_box),
        top_k=top_k * 4 if prefix_model is not None else top_k,
        max_pair_candidates=int(st.session_state.max_pair_candidates),
    )
    return rerank_my_plans(
        plans,
        prefix_model,
        relation_model,
        {
            "my_picks": my_picks,
            "enemy_picks": enemy_picks,
            "turn": turn,
            "first_pick_side": first_pick_side,
            "my_leader": st.session_state.my_leader,
            "enemy_leader": st.session_state.enemy_leader,
            "my_box": list(st.session_state.my_box),
            "use_my_box": bool(st.session_state.use_my_box),
            "max_pair_candidates": int(st.session_state.max_pair_candidates),
        },
        assets,
        top_k,
    )


def calculate_enemy_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    turn = len(my_picks) + len(enemy_picks) + 1
    prefix_model = get_prefix_model(assets)
    relation_model = get_relation_model(assets)
    top_k = int(st.session_state.top_k)
    predictions = relation_model.recommend_enemy_next_picks(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        first_pick_side=st.session_state.first_pick_side,
        top_k=top_k * 4 if prefix_model is not None else top_k,
        candidate_pool=assets["options"],
        max_pair_candidates=int(st.session_state.max_pair_candidates),
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
    )
    return rerank_enemy_plans(
        predictions,
        prefix_model,
        relation_model,
        {
            "my_picks": my_picks,
            "enemy_picks": enemy_picks,
            "turn": turn,
            "first_pick_side": st.session_state.first_pick_side,
            "my_leader": st.session_state.my_leader,
            "enemy_leader": st.session_state.enemy_leader,
            "my_box": list(st.session_state.my_box),
            "use_my_box": bool(st.session_state.use_my_box),
            "max_pair_candidates": int(st.session_state.max_pair_candidates),
        },
        assets,
        top_k,
    )


def calculate_ban_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    top_k = int(st.session_state.top_k)
    ban_model = get_ban_model(assets)
    if ban_model is not None:
        return ban_model.recommend_bans(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            my_leader=st.session_state.my_leader,
            enemy_leader=st.session_state.enemy_leader,
            top_k=top_k,
        )

    relation_model = get_relation_model(assets)
    return relation_model.recommend_ban_plans(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        top_k=top_k,
        first_pick_side=st.session_state.first_pick_side,
    )


def filter_gallery_monsters(assets, search_text, attr_filter, family_filter, only_owned, only_with_image):
    monsters = list(assets["options"])
    if only_owned:
        owned = set(st.session_state.my_box)
        monsters = [monster for monster in monsters if monster in owned]

    search_text = str(search_text or "").strip().lower()
    if search_text:
        monsters = [
            monster
            for monster in monsters
            if search_text in assets["search_text_map"].get(monster, monster.lower())
        ]
    if attr_filter != "全部":
        monsters = [monster for monster in monsters if assets["attr_map"].get(monster) == attr_filter]
    if family_filter != "全部":
        monsters = [monster for monster in monsters if assets["family_map"].get(monster) == family_filter]
    if only_with_image:
        monsters = [
            monster
            for monster in monsters
            if get_image_path(monster, assets["image_lookup"])
        ]

    occupied = set(st.session_state.my_picks) | set(st.session_state.enemy_picks)
    monsters = [monster for monster in monsters if monster not in occupied]
    return monsters


def build_picker_payload(monsters, assets):
    state_names = list(dict.fromkeys(
        list(monsters)
        + list(st.session_state.my_picks)
        + list(st.session_state.enemy_picks)
    ))
    state_items = {}
    for monster in state_names:
        image_path = get_image_path(monster, assets["image_lookup"])
        state_items[monster] = {
            "name": monster,
            "attr": assets["attr_map"].get(monster, ""),
            "family": assets["family_map"].get(monster, ""),
            "image": load_image_data_uri(str(image_path)) if image_path else "",
        }

    return [
        state_items[monster]
        for monster in monsters
        if monster in state_items
    ], state_items


def clean_picker_values(values, options):
    cleaned = []
    seen = set()
    for value in values or []:
        monster = str(value or "")
        if monster in options and monster not in seen:
            cleaned.append(monster)
            seen.add(monster)
        if len(cleaned) >= 5:
            break
    return cleaned


def apply_picker_value(value, assets):
    if not isinstance(value, dict):
        return False
    event_id = str(value.get("event_id", "") or "")
    if not event_id or event_id == st.session_state.picker_event_id:
        return False

    options = set(assets["options"])
    my_picks = clean_picker_values(value.get("my_picks"), options)
    enemy_picks = clean_picker_values(value.get("enemy_picks"), options)
    my_set = set(my_picks)
    enemy_picks = [monster for monster in enemy_picks if monster not in my_set]

    st.session_state.my_picks = my_picks
    st.session_state.enemy_picks = enemy_picks
    st.session_state.my_leader = (
        str(value.get("my_leader", "") or "")
        if str(value.get("my_leader", "") or "") in my_picks
        else ""
    )
    st.session_state.enemy_leader = (
        str(value.get("enemy_leader", "") or "")
        if str(value.get("enemy_leader", "") or "") in enemy_picks
        else ""
    )
    st.session_state.picker_event_id = event_id
    clear_results()
    return True


@st.fragment
def render_bp_picker_fragment(assets):
    control_columns = st.columns([1.4, 1, 1.35, 1.45])
    with control_columns[0]:
        st.radio(
            "先手",
            options=("me", "enemy"),
            format_func=lambda value: "我方一选" if value == "me" else "对方一选",
            key="first_pick_side",
            horizontal=True,
        )
    with control_columns[1]:
        st.checkbox("只看我的 Box", key="gallery_only_owned")
    with control_columns[2]:
        st.selectbox("属性", options=assets["attributes"], key="gallery_attr")
    with control_columns[3]:
        st.selectbox("系列", options=assets["families"], key="gallery_family")
    st.text_input(
        "搜索魔灵",
        key="gallery_search",
        placeholder="输入名称筛选图标",
        label_visibility="collapsed",
    )
    st.checkbox("只显示有图片", key="gallery_only_with_image")

    gallery_monsters = filter_gallery_monsters(
        assets,
        st.session_state.gallery_search,
        st.session_state.gallery_attr,
        st.session_state.gallery_family,
        st.session_state.gallery_only_owned,
        st.session_state.gallery_only_with_image,
    )
    total_pages = max(1, (len(gallery_monsters) + BP_GALLERY_PAGE_SIZE - 1) // BP_GALLERY_PAGE_SIZE)
    st.session_state.gallery_page = max(1, min(int(st.session_state.gallery_page), total_pages))
    start = (st.session_state.gallery_page - 1) * BP_GALLERY_PAGE_SIZE
    page_items = gallery_monsters[start : start + BP_GALLERY_PAGE_SIZE]
    page_columns = st.columns([1, 2, 1])
    with page_columns[0]:
        if st.button("上一页", key="gallery_prev", use_container_width=True):
            st.session_state.gallery_page = max(1, st.session_state.gallery_page - 1)
            st.rerun(scope="fragment")
    with page_columns[1]:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem'>第 {st.session_state.gallery_page}/{total_pages} 页 · 共 {len(gallery_monsters)} 个</div>",
            unsafe_allow_html=True,
        )
    with page_columns[2]:
        if st.button("下一页", key="gallery_next", use_container_width=True):
            st.session_state.gallery_page = min(total_pages, st.session_state.gallery_page + 1)
            st.rerun(scope="fragment")

    items, state_items = build_picker_payload(page_items, assets)
    current_turn = len(st.session_state.my_picks) + len(st.session_state.enemy_picks) + 1
    current_side = side_for_turn(current_turn, st.session_state.first_pick_side) or "none"
    picker_value = BP_ICON_PICKER(
        items=items,
        state_items=state_items,
        current_turn=current_turn,
        current_side=current_side,
        phase=phase_label(
            st.session_state.my_picks,
            st.session_state.enemy_picks,
            current_side,
        ),
        first_pick_side=st.session_state.first_pick_side,
        my_picks=st.session_state.my_picks,
        enemy_picks=st.session_state.enemy_picks,
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
        key="bp_icon_picker",
        default=None,
    )
    if apply_picker_value(picker_value, assets):
        st.rerun(scope="fragment")

    current_turn = len(st.session_state.my_picks) + len(st.session_state.enemy_picks) + 1
    current_side = side_for_turn(current_turn, st.session_state.first_pick_side)
    signature = make_signature(
        st.session_state.my_picks,
        st.session_state.enemy_picks,
        st.session_state.first_pick_side,
        st.session_state.my_leader,
        st.session_state.enemy_leader,
    )

    action_columns = st.columns(4)
    with action_columns[0]:
        if st.button("清空 BP 状态", key="clear_bp_fragment", use_container_width=True):
            st.session_state.my_picks = []
            st.session_state.enemy_picks = []
            st.session_state.my_leader = ""
            st.session_state.enemy_leader = ""
            clear_results()
            st.rerun(scope="fragment")
    with action_columns[1]:
        if st.button(
            "生成我方推荐",
            key="pick_recommendation_fragment",
            type="primary",
            use_container_width=True,
            disabled=current_side != "me" or len(st.session_state.my_picks) >= 5,
        ):
            try:
                with st.spinner("正在计算我方推荐..."):
                    st.session_state.pick_results = {
                        "signature": signature,
                        "items": calculate_my_recommendations(assets),
                    }
            except Exception as exc:
                st.session_state.pick_results = {"signature": signature, "error": str(exc), "items": []}
    with action_columns[2]:
        if st.button(
            "预测对方下一选",
            key="enemy_recommendation_fragment",
            use_container_width=True,
            disabled=current_side != "enemy" or len(st.session_state.enemy_picks) >= 5,
        ):
            try:
                with st.spinner("正在预测对方下一选..."):
                    st.session_state.enemy_results = {
                        "signature": signature,
                        "items": calculate_enemy_recommendations(assets),
                    }
            except Exception as exc:
                st.session_state.enemy_results = {"signature": signature, "error": str(exc), "items": []}
    with action_columns[3]:
        if st.button(
            "推荐 Ban",
            key="ban_recommendation_fragment",
            use_container_width=True,
            disabled=len(st.session_state.my_picks) != 5 or len(st.session_state.enemy_picks) != 5,
        ):
            try:
                with st.spinner("正在计算 Ban..."):
                    st.session_state.ban_results = {
                        "signature": signature,
                        "items": calculate_ban_recommendations(assets),
                    }
            except Exception as exc:
                st.session_state.ban_results = {"signature": signature, "error": str(exc), "items": []}

    for title, key, result_type in (
        ("我方推荐", "pick_results", "pick"),
        ("对方下一选预测", "enemy_results", "enemy"),
        ("Ban 推荐", "ban_results", "ban"),
    ):
        result = st.session_state.get(key)
        if not result or result.get("signature") != signature:
            continue
        if result.get("error"):
            st.error(result["error"])
            continue
        render_results(title, result.get("items", []), assets, result_type)


def toggle_my_box(monster):
    if monster in st.session_state.my_box:
        st.session_state.my_box = [
            item for item in st.session_state.my_box
            if item != monster
        ]
    else:
        st.session_state.my_box = sorted(
            set(st.session_state.my_box + [monster]),
            key=normalize_name,
        )


def render_box_monsters(monsters, assets):
    for row_start in range(0, len(monsters), 2):
        columns = st.columns(2)
        for column, monster in zip(columns, monsters[row_start : row_start + 2]):
            with column:
                image_path = get_image_path(monster, assets["image_lookup"])
                if image_path:
                    image_uri = html.escape(
                        load_image_data_uri(str(image_path)),
                        quote=True,
                    )
                    st.markdown(
                        f"""
                        <div class="box-monster-preview">
                            <img src="{image_uri}" alt="{html.escape(str(monster), quote=True)}">
                            <span>{html.escape(str(monster))}</span>
                            <small>{html.escape(str(assets["attr_map"].get(monster, "") or ""))}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if st.button(
                    "移出 Box" if monster in st.session_state.my_box else "加入 Box",
                    key=f"box_toggle_{normalize_name(monster)}",
                    type="primary" if monster in st.session_state.my_box else "secondary",
                    use_container_width=True,
                ):
                    toggle_my_box(monster)
                    st.rerun(scope="fragment")


@st.fragment
def render_box_fragment(assets):
    st.subheader("我的 Box")
    st.text_input(
        "搜索 Box 图标",
        key="box_search",
        label_visibility="collapsed",
        placeholder="搜索魔灵或中文别称",
    )
    monsters = list(assets["options"])
    search_text = str(st.session_state.box_search or "").strip().lower()
    if search_text:
        monsters = [
            monster
            for monster in monsters
            if search_text in assets["search_text_map"].get(monster, monster.lower())
        ]
    monsters = [
        monster
        for monster in monsters
        if get_image_path(monster, assets["image_lookup"])
    ]
    selected = set(st.session_state.my_box)
    selected_monsters = [monster for monster in monsters if monster in selected]
    available_monsters = [monster for monster in monsters if monster not in selected]

    st.caption(
        f"已在 Box {len(selected_monsters)} 个 · "
        f"未选 {len(available_monsters)} 个 · 共 {len(monsters)} 个"
    )
    st.markdown(
        f'<div class="box-section-title">已在 Box ({len(selected_monsters)})</div>',
        unsafe_allow_html=True,
    )
    if selected_monsters:
        render_box_monsters(selected_monsters, assets)
    else:
        st.caption("还没有选择魔灵。")

    st.markdown(
        f'<div class="box-section-title">未在 Box ({len(available_monsters)})</div>',
        unsafe_allow_html=True,
    )
    page_size = BP_GALLERY_PAGE_SIZE
    total_pages = max(1, (len(available_monsters) + page_size - 1) // page_size)
    st.session_state.box_page = max(1, min(int(st.session_state.box_page), total_pages))
    start = (st.session_state.box_page - 1) * page_size
    page_items = available_monsters[start : start + page_size]
    render_box_monsters(page_items, assets)

    box_nav_left, box_nav_label, box_nav_right = st.columns([1, 2, 1])
    with box_nav_left:
        if st.button("上一页", key="box_prev", use_container_width=True):
            st.session_state.box_page = max(1, st.session_state.box_page - 1)
            st.rerun(scope="fragment")
    with box_nav_label:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem'>第 {st.session_state.box_page} 页</div>",
            unsafe_allow_html=True,
        )
    with box_nav_right:
        if st.button("下一页", key="box_next", use_container_width=True):
            st.session_state.box_page = min(total_pages, st.session_state.box_page + 1)
            st.rerun(scope="fragment")

    save_column, clear_column = st.columns(2)
    with save_column:
        if st.button("保存 Box", key="save_box", use_container_width=True):
            save_my_box()
            st.success("已保存")
    with clear_column:
        if st.button("清空 Box", key="clear_box", use_container_width=True):
            st.session_state.my_box = []
            st.rerun(scope="fragment")


def render_result_card(index, item, assets, result_type):
    if result_type == "enemy":
        monsters = [value for value in item.get("plan") or item.get("monsters") or [] if value]
        if not monsters:
            monsters = [item.get("monster") or item.get("display_name") or ""]
        score = item.get("final_score", item.get("plan_score", 0.0))
    elif result_type == "ban":
        monsters = [item.get("ban") or item.get("monster") or ""]
        score = item.get("plan_score", item.get("score", 0.0))
    else:
        monsters = [value for value in item.get("plan", []) if value]
        score = item.get("plan_score", 0.0)

    detail = item.get("detail", {}) or {}
    win_rate = item.get("estimated_win_rate")
    if win_rate is not None and float(win_rate or 0.0) > 0.0:
        win_rate_text = f"{float(win_rate):.1%}"
    else:
        win_rate_text = "未启用"

    if "win_rate_gain" in detail:
        secondary_label = "变化"
        secondary_value = f"{float(detail['win_rate_gain']):+.1%}"
    elif "post_enemy_pick_win_rate" in detail:
        secondary_label = "我方胜率"
        secondary_value = f"{float(detail['post_enemy_pick_win_rate']):.1%}"
    elif "post_ban_win_rate" in detail:
        secondary_label = "Ban 后"
        secondary_value = f"{float(detail['post_ban_win_rate']):.1%}"
    else:
        secondary_label = "动作"
        secondary_value = str(detail.get("action_type") or item.get("source", "relation"))

    ranking_source = detail.get("ranking_source") or item.get("source") or "relation"
    source_text = {
        "prefix_value_0.65_relation_0.25_stats_0.10": "Value + Relation + Stats",
        "prefix_value_0.70_relation_0.30": "Value + Relation",
        "prefix_value_0.55_relation_0.20_stats_0.10_mechanism_0.15": "Value + Relation + Stats + Mechanism",
        "prefix_value_0.60_relation_0.25_mechanism_0.15": "Value + Relation + Mechanism",
        "enemy_value_0.65_relation_0.25_stats_0.10": "Enemy Value + Stats",
        "enemy_value_0.70_relation_0.30": "Enemy Value + Relation",
        "enemy_value_0.55_relation_0.20_stats_0.10_mechanism_0.15": "Enemy Value + Stats + Mechanism",
        "enemy_value_0.60_relation_0.25_mechanism_0.15": "Enemy Value + Relation + Mechanism",
        "ban_value_0.75_mechanism_0.25": "Ban Value + Mechanism",
        "ban_value_0.75_mechanism_0.25_stats_0.10": "Ban Value + Mechanism + Stats",
        "ban_value_0.75_stats_0.10": "Ban Value + Stats",
        "ban_value_0.75": "Ban Value",
        "minimax_value_0.60_immediate_0.15_relation_0.15_evidence_0.10": "两步预演 + Value + 数据",
        "minimax_enemy_0.60_best_response_0.15_immediate_0.15_relation_0.10_evidence": "对手预演 + 反制价值",
        "relation": "Relation",
    }.get(str(ranking_source), str(ranking_source))
    safe_title = html.escape(f"{index}. {' + '.join(monsters)}")
    image_markup = []
    for monster in monsters:
        image_path = get_image_path(monster, assets["image_lookup"])
        if image_path:
            image_uri = html.escape(load_image_data_uri(str(image_path)), quote=True)
            image_markup.append(
                f'<div class="recommendation-monster">'
                f'<img src="{image_uri}" alt="{html.escape(str(monster), quote=True)}">'
                f'<span>{html.escape(str(monster))}</span></div>'
            )
        else:
            image_markup.append(
                f'<div class="recommendation-monster">'
                f'<div class="recommendation-placeholder">?</div>'
                f'<span>{html.escape(str(monster))}</span></div>'
            )
    reasons = item.get("reasons", [])
    reason_markup = (
        f'<div class="recommendation-reason">{html.escape("；".join(str(reason) for reason in reasons))}</div>'
        if reasons
        else ""
    )
    card_markup = f"""
        <div class="recommendation-card">
            <div class="recommendation-header">
                <strong>{safe_title}</strong>
                <div class="recommendation-actions">
                    <span class="recommendation-score">{float(score or 0.0):.3f}</span>
                </div>
            </div>
            <div class="recommendation-monsters">{''.join(image_markup)}</div>
            <div class="recommendation-metrics">
                <div><span>估计胜率</span><b>{win_rate_text}</b></div>
                <div><span>{html.escape(secondary_label)}</span><b>{html.escape(secondary_value)}</b></div>
                <div><span>来源</span><b>{html.escape(source_text)}</b></div>
            </div>
            {reason_markup}
        </div>
    """
    st.markdown(card_markup, unsafe_allow_html=True)
    if result_type in {"pick", "enemy"}:
        if st.button(
            "采用",
            key=f"recommendation_apply_{result_type}_{index}",
            type="primary",
            use_container_width=True,
        ):
            apply_recommendation_plan(result_type, monsters, set(assets["options"]))
            st.rerun(scope="fragment")
    if st.session_state.show_details and detail:
        with st.expander("模型明细", expanded=False):
            st.json(detail)


def render_results(title, results, assets, result_type):
    st.subheader(title)
    if not results:
        st.info("没有可显示的推荐结果。")
        return
    for start in range(0, len(results), 3):
        result_row = results[start : start + 3]
        columns = st.columns(3)
        for column, (index, item) in zip(
            columns,
            enumerate(result_row, start=start + 1),
        ):
            with column:
                render_result_card(index, item, assets, result_type)


def render_model_status(assets):
    bundle = assets["relation_bundle"]
    if bundle:
        st.success(f"Relation：{bundle['label']}")
    else:
        st.error("Relation：未找到模型")
    if assets["prefix_value_ready"]:
        st.success("Prefix Value：已加载")
    else:
        st.warning("Prefix Value：未训练，选人排序使用 Relation")
    if assets["ban_value_ready"]:
        st.success("Ban Value：已加载")
    else:
        st.warning("Ban Value：未找到，完整阵容 Ban 使用 Relation 备用")


def main():
    st.set_page_config(page_title="BP Decision Workbench", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] { background: #f7f9fc; border: 1px solid #e5e9f0; padding: 0.65rem; }
        div[data-testid="stVerticalBlock"] > div:has(> div.stButton) button { min-height: 2.4rem; }
        .recommendation-card {
            min-height: 144px;
            margin-bottom: 0.45rem;
            padding: 0.55rem 0.6rem 0.5rem;
            overflow: hidden;
            background: #ffffff;
            border: 1px solid #d9e0ea;
            border-radius: 6px;
        }
        .recommendation-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.35rem;
            min-width: 0;
            color: #1f2937;
            font-size: 0.8rem;
            line-height: 1.15;
        }
        .recommendation-header strong {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .recommendation-score {
            flex: 0 0 auto;
            color: #2457a6;
            font-size: 0.86rem;
            font-variant-numeric: tabular-nums;
        }
        .recommendation-actions {
            display: flex;
            flex: 0 0 auto;
            align-items: center;
            gap: 0.35rem;
        }
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
            min-height: 3rem !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .recommendation-monsters {
            display: flex;
            min-height: 70px;
            align-items: start;
            justify-content: center;
            gap: 0.35rem;
            margin: 0.45rem 0 0.35rem;
        }
        .recommendation-monster {
            min-width: 0;
            flex: 1 1 0;
            text-align: center;
        }
        .recommendation-monster img,
        .recommendation-placeholder {
            display: block;
            width: 54px;
            height: 54px;
            margin: 0 auto 0.15rem;
            object-fit: cover;
            background: #f3f5f8;
            border-radius: 4px;
        }
        .recommendation-placeholder {
            color: #8a94a6;
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 54px;
        }
        .recommendation-monster span {
            display: block;
            overflow: hidden;
            color: #4b5563;
            font-size: 0.61rem;
            line-height: 1.05;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .recommendation-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.25rem;
            border-top: 1px solid #edf0f5;
            padding-top: 0.35rem;
        }
        .recommendation-metrics div {
            min-width: 0;
            text-align: center;
        }
        .recommendation-metrics span,
        .recommendation-metrics b {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .recommendation-metrics span {
            color: #7b8798;
            font-size: 0.59rem;
            line-height: 1.05;
        }
        .recommendation-metrics b {
            margin-top: 0.1rem;
            color: #27364d;
            font-size: 0.68rem;
            font-weight: 650;
            line-height: 1.1;
        }
        .recommendation-reason {
            margin-top: 0.3rem;
            overflow: hidden;
            color: #748094;
            font-size: 0.59rem;
            line-height: 1.1;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .monster-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(82px, 1fr));
            gap: 7px;
            align-items: stretch;
        }
        .box-monster-preview {
            min-height: 84px;
            margin-bottom: 0.2rem;
            padding: 0.25rem;
            overflow: hidden;
            text-align: center;
            background: #f7f9fc;
            border: 1px solid #d9e0ea;
            border-radius: 5px;
        }
        .box-monster-preview img {
            display: block;
            width: 62px;
            height: 62px;
            margin: 0 auto 0.15rem;
            object-fit: cover;
            border-radius: 4px;
        }
        .box-monster-preview span,
        .box-monster-preview small {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .box-monster-preview span {
            color: #273142;
            font-size: 0.64rem;
            line-height: 1.1;
        }
        .box-monster-preview small {
            margin-top: 0.1rem;
            color: #748094;
            font-size: 0.58rem;
            line-height: 1.05;
        }
        .box-section-title {
            margin: 0.65rem 0 0.35rem;
            padding-top: 0.45rem;
            color: #27364d;
            border-top: 1px solid #d9e0ea;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .monster-card {
            display: flex;
            min-width: 0;
            min-height: 112px;
            flex-direction: column;
            align-items: center;
            padding: 5px 4px 4px;
            color: #1f2937 !important;
            background: #ffffff;
            border: 1px solid #d9e0ea;
            border-radius: 6px;
            text-decoration: none !important;
            transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
        }
        .monster-card:hover {
            border-color: #4b72c2;
            box-shadow: 0 2px 8px rgba(43, 72, 125, 0.16);
            transform: translateY(-1px);
        }
        .monster-card.is-selected {
            border-color: #2f855a;
            background: #f0faf4;
        }
        .monster-icon {
            width: 68px;
            height: 68px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: #f3f5f8;
            border-radius: 4px;
        }
        .monster-icon img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .monster-icon-placeholder, .state-icon-placeholder {
            color: #8a94a6;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .monster-name {
            width: 100%;
            min-height: 2.25em;
            margin-top: 3px;
            overflow-wrap: anywhere;
            text-align: center;
            font-size: 0.72rem;
            line-height: 1.12;
        }
        .monster-subtitle {
            width: 100%;
            min-height: 1.05em;
            overflow: hidden;
            color: #748094;
            text-align: center;
            font-size: 0.63rem;
            line-height: 1.05;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        .state-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 5px;
            margin: 5px 0 12px;
        }
        .state-slot {
            min-width: 0;
            padding: 4px 2px 3px;
            text-align: center;
            background: #ffffff;
            border: 1px solid #d9e0ea;
            border-radius: 5px;
        }
        .state-slot.empty {
            min-height: 106px;
            color: #8a94a6;
            background: #f7f9fc;
        }
        .state-slot-remove {
            display: block;
            color: #1f2937 !important;
            text-decoration: none !important;
        }
        .state-slot-number {
            color: #748094;
            font-size: 0.64rem;
            line-height: 1.1;
        }
        .state-icon {
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 2px auto;
            overflow: hidden;
            background: #f3f5f8;
            border-radius: 4px;
        }
        .state-icon img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .state-name {
            min-height: 2.2em;
            overflow-wrap: anywhere;
            font-size: 0.64rem;
            line-height: 1.1;
        }
        .state-slot-empty {
            display: flex;
            min-height: 78px;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
        }
        .leader-link {
            display: block;
            margin-top: 3px;
            color: #748094 !important;
            font-size: 0.62rem;
            text-decoration: none !important;
        }
        .leader-link:hover, .leader-link.leader-active {
            color: #b7791f !important;
            font-weight: 700;
        }
        @media (max-width: 700px) {
            .recommendation-card { min-height: 136px; padding: 0.45rem; }
            .recommendation-monsters { min-height: 62px; }
            .recommendation-monster img,
            .recommendation-placeholder { width: 46px; height: 46px; line-height: 46px; }
            .monster-grid { grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); gap: 5px; }
            .monster-card { min-height: 102px; }
            .monster-icon { width: 58px; height: 58px; }
            .state-grid { gap: 3px; }
            .state-icon { width: 42px; height: 42px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    assets = load_web_assets()
    initialize_state(assets)
    sample_count = load_sample_count(str(SAMPLES_DB_PATH))

    with st.sidebar:
        st.header("控制")
        st.checkbox("只从我的 Box 推荐", key="use_my_box")
        st.slider("结果数量", min_value=1, max_value=12, key="top_k")
        st.slider(
            "双选候选预筛选",
            min_value=0,
            max_value=500,
            step=10,
            key="max_pair_candidates",
        )
        st.checkbox("显示模型明细", key="show_details")
        st.selectbox(
            "推理设备",
            options=("auto", "cpu", "cuda"),
            format_func=current_device_label,
            key="device",
        )
        if st.button("重新载入模型", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        render_box_fragment(assets)

        st.divider()
        st.subheader("运行状态")
        render_model_status(assets)
        st.caption(f"逐回合样本：{sample_count:,}")
        st.caption(f"推理设备：{st.session_state.device}")

    # st.title("BP Decision Workbench")
    # st.caption("点击左侧魔灵图标选人，BP 槽位在浏览器内即时更新，不刷新整页")
    render_bp_picker_fragment(assets)


if __name__ == "__main__":
    main()
