from __future__ import annotations

import base64
import html
import json
import mimetypes
import sqlite3
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from bp_core.account_profile import load_account_profile
from bp_core.action_space import timeline_from_picks
from bp_core.draft_state import DraftState
from bp_recommender.decision_engine import BPDecisionEngine
from bp_core.lucksack_stats import (
    BPLucksackStats,
    build_catalog_ambiguous_short_names,
    is_usable_catalog_item,
)
from bp_core.bp_common import infer_pick_count, infer_pick_count_for_side, side_for_order
from bp_recommender.legacy_adapter import LegacyBPRecommender
from bp_core.names import normalize_name, slugify_name
from streamlit.errors import StreamlitInvalidLayoutContextError


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
ACCOUNT_PROFILE_PATH = JSON_DIR / "account_profile.json"
IMAGE_MAP_PATH = JSON_DIR / "monster_images.json"
ALIAS_MAP_PATH = JSON_DIR / "monster_alias_map.json"
ATTR_MAP_PATH = JSON_DIR / "monster_attributes.json"
FAMILY_MAP_PATH = JSON_DIR / "monster_families.json"
SAMPLES_DB_PATH = DATA_DIR / "bp_samples.sqlite"
LUCK_STATS_PATH = JSON_DIR / "lucksack_stats.json"
LUCK_CATALOG_PATH = JSON_DIR / "lucksack_monster_catalog.json"
SWKIT_ALIAS_PATH = JSON_DIR / "swkit_monster_aliases.json"

SHARED_BUNDLE = (
    "Legacy Relation + Prefix Value + Ban Value",
    MODEL_DIR / "bp_relation_net_v2.pt",
    MODEL_DIR / "bp_relation_meta_v2.json",
    MODEL_DIR / "bp_relation_vocab_v2.json",
)
LEGACY_VALUE_ARTIFACTS = (
    MODEL_DIR / "bp_prefix_value_net.pt",
    MODEL_DIR / "bp_prefix_value_meta.json",
    MODEL_DIR / "bp_prefix_value_vocab.json",
    MODEL_DIR / "bp_draft_value_net.pt",
    MODEL_DIR / "bp_draft_value_meta.json",
    MODEL_DIR / "bp_draft_value_vocab.json",
)

BP_GALLERY_COLUMNS = 17
BP_GALLERY_PAGE_ROWS = 3
BP_GALLERY_PAGE_SIZE = BP_GALLERY_COLUMNS * BP_GALLERY_PAGE_ROWS


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def bundle_ready(bundle):
    return all(Path(path).exists() for path in bundle)


def rerun_fragment_or_app():
    try:
        st.rerun(scope="fragment")
    except StreamlitInvalidLayoutContextError:
        st.rerun()


def asset_cache_stamp():
    paths = (
        IMAGE_MAP_PATH,
        ALIAS_MAP_PATH,
        ATTR_MAP_PATH,
        FAMILY_MAP_PATH,
        LUCK_STATS_PATH,
        LUCK_CATALOG_PATH,
        SWKIT_ALIAS_PATH,
        MY_BOX_PATH,
        *SHARED_BUNDLE[1:],
        *LEGACY_VALUE_ARTIFACTS,
    )
    return tuple(
        (
            str(path),
            int(path.stat().st_mtime_ns),
            int(path.stat().st_size),
        )
        for path in paths
        if path.exists()
    )


def select_shared_bundle():
    label, model_path, meta_path, vocab_path = SHARED_BUNDLE
    if not bundle_ready((model_path, meta_path, vocab_path, *LEGACY_VALUE_ARTIFACTS)):
        return None
    return {
        "label": label,
        "model_path": model_path,
        "meta_path": meta_path,
        "vocab_path": vocab_path,
    }


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
        normalize_key(slugify_name(text)),
    }


def lucksack_image_url(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return f"https://static.lucksack.gg/images/monsters/{raw.lstrip('/')}"


def resolve_option_name(value, assets):
    text = str(value or "").strip()
    if not text:
        return ""
    alias_lookup = assets.get("alias_lookup") or {}
    for variant in value_variants(text):
        resolved = alias_lookup.get(variant)
        if resolved:
            return resolved
    options = assets.get("options") or []
    normalized = normalize_name(text)
    for option in options:
        if normalize_name(option) == normalized:
            return option
    return ""


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


def build_monster_options(vocab, display_name_map, frequency_map):
    options = []
    seen = set()
    vocab_items = []
    for slug, raw_id in (vocab or {}).items() if isinstance(vocab, dict) else ():
        try:
            vocab_items.append((slug, int(raw_id)))
        except (TypeError, ValueError):
            continue
    for slug, raw_id in sorted(vocab_items, key=lambda item: item[1]):
        if raw_id <= 1:
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
            frequency = float(raw_frequency or 0)
        except (TypeError, ValueError):
            frequency = 0.0
        options.append((display_name, frequency))

    unique = {}
    for name, frequency in options:
        unique.setdefault(normalize_name(name), (name, frequency))
    return [
        name
        for name, _frequency in sorted(
            unique.values(),
            key=lambda item: (-float(item[1]), normalize_name(item[0])),
        )
    ]


def matches_monster_search(search_text, searchable_text):
    """Match a query against names and aliases with the same normalization."""
    query_variants = value_variants(search_text)
    if not query_variants:
        return True
    searchable_variants = value_variants(searchable_text)
    return any(
        query in candidate
        for query in query_variants
        for candidate in searchable_variants
    )


@st.cache_data(show_spinner=False)
def load_web_assets(asset_stamp=None):
    shared_bundle = select_shared_bundle()
    shared_meta = read_json(shared_bundle["meta_path"], {}) if shared_bundle else {}
    shared_vocab = read_json(shared_bundle["vocab_path"], {}) if shared_bundle else {}
    image_map = read_json(IMAGE_MAP_PATH, {})
    generated_alias_map = read_json(ALIAS_MAP_PATH, {})
    if not isinstance(generated_alias_map, dict):
        generated_alias_map = {}
    attr_map = read_json(ATTR_MAP_PATH, {})
    family_map = read_json(FAMILY_MAP_PATH, {})
    if not isinstance(shared_meta, dict):
        shared_meta = {}
    if not isinstance(shared_vocab, dict):
        shared_vocab = {}
    if not isinstance(image_map, dict):
        image_map = {}
    if not isinstance(attr_map, dict):
        attr_map = {}
    if not isinstance(family_map, dict):
        family_map = {}
    luck_stats = BPLucksackStats(stats_path=LUCK_STATS_PATH, catalog_path=LUCK_CATALOG_PATH)
    swkit_alias_data = read_json(SWKIT_ALIAS_PATH, {})
    if not isinstance(swkit_alias_data, dict):
        swkit_alias_data = {}
    my_box = read_json(MY_BOX_PATH, [])
    account_profile = load_account_profile(ACCOUNT_PROFILE_PATH)
    if not isinstance(my_box, list):
        my_box = []

    remote_catalog = luck_stats.catalog if isinstance(luck_stats.catalog, list) else []
    remote_image_map = {}
    remote_names = set()
    alias_sets = {}
    catalog_slugs = set()
    usable_catalog = [
        item for item in remote_catalog
        if is_usable_catalog_item(item)
    ]
    ambiguous_short_names = build_catalog_ambiguous_short_names(remote_catalog)
    usable_catalog_keys = set()
    for item in usable_catalog:
        for value in (
            item.get("slug"),
            item.get("label"),
            luck_stats.to_display_name(item.get("slug")),
        ):
            usable_catalog_keys.update(value_variants(value))

    def add_aliases(display_name, *values):
        display_name = str(display_name or "").strip()
        if not display_name:
            return
        aliases = alias_sets.setdefault(display_name, set())
        for value in values:
            if isinstance(value, (list, tuple, set)):
                values_to_add = value
            else:
                values_to_add = (value,)
            for alias in values_to_add:
                alias = str(alias or "").strip()
                if alias:
                    aliases.add(alias)

    for item in remote_catalog:
        if not is_usable_catalog_item(item):
            continue
        catalog_slugs.add(normalize_name(item.get("slug")))
        display_name = luck_stats.to_display_name(item.get("slug"))
        if not display_name:
            continue
        remote_names.add(display_name)
        add_aliases(
            display_name,
            item.get("slug"),
            item.get("label"),
            item.get("com2us_id"),
        )
        unit_id = str(item.get("com2us_id") or "")
        aliases = (swkit_alias_data.get("aliases_by_id") or {}).get(unit_id, [])
        add_aliases(display_name, aliases)
        image_name = str(item.get("image") or "").strip()
        if image_name:
            remote_image_map[display_name] = lucksack_image_url(image_name)
        attr_map.setdefault(display_name, str(item.get("element") or ""))
        family = str(
            item.get("family")
            or item.get("family_name")
            or item.get("familyName")
            or ""
        ).strip()
        if family:
            family_map.setdefault(display_name, family)

    for slug in luck_stats.candidate_slugs():
        if slug in catalog_slugs:
            continue
        display_name = luck_stats.to_display_name(slug)
        if not display_name:
            continue
        if normalize_name(display_name) in ambiguous_short_names:
            continue
        remote_names.add(display_name)
        add_aliases(display_name, slug)
        image_name = luck_stats.to_image_name(slug)
        if image_name:
            remote_image_map[display_name] = lucksack_image_url(image_name)

    for alias, display_name in (generated_alias_map or {}).items():
        add_aliases(display_name, alias)
        remote_names.add(str(display_name).strip())

    frequency_map = dict(shared_meta.get("monster_frequency", {}) or {})
    for slug in luck_stats.candidate_slugs():
        frequency = luck_stats.pick_frequency(slug)
        try:
            current_frequency = float(frequency_map.get(slug, 0) or 0)
        except (TypeError, ValueError):
            current_frequency = 0.0
        if frequency > current_frequency:
            frequency_map[slug] = frequency

    options = build_monster_options(
        shared_vocab,
        shared_meta.get("display_name_map", {}) or {},
        frequency_map,
    )
    if options and usable_catalog_keys:
        options = [
            option
            for option in options
            if value_variants(option) & usable_catalog_keys
        ]
    model_options = list(options)
    if not options:
        # The catalog is also the source for the picker and Box editor.  It
        # must remain usable before the legacy model bundle has been trained; the
        # recommender itself will still report that its model bundle is absent.
        options = sorted(
            remote_names,
            key=lambda name: (
                -luck_stats.pick_frequency(luck_stats.resolve_slug(name)),
                normalize_name(name),
            ),
        )
    gallery_by_key = {}
    for name in sorted(
        set(remote_names) | set(model_options),
        key=lambda value: (
            normalize_name(value),
            str(value).islower(),
            -len(str(value)),
            str(value),
        ),
    ):
        gallery_by_key.setdefault(normalize_name(name), name)
    gallery_options = sorted(
        gallery_by_key.values(),
        key=lambda name: (
            -luck_stats.pick_frequency(luck_stats.resolve_slug(name)),
            normalize_name(name),
        ),
    )
    model_options = list(
        dict.fromkeys(
            gallery_by_key.get(normalize_name(name), name)
            for name in model_options
        )
    )
    if not gallery_options:
        gallery_options = list(options)
    option_set = set(gallery_options)
    if not my_box and account_profile is not None:
        owned_slugs = set(account_profile.owned_monsters)
        my_box = [
            name
            for name in gallery_options
            if normalize_name(name) in owned_slugs
        ]
    my_box = [name for name in my_box if name in option_set]
    option_lookup = {}
    for option in gallery_options:
        for variant in value_variants(option):
            option_lookup.setdefault(variant, option)
    alias_targets = {}
    for display_name, aliases in alias_sets.items():
        target = ""
        for variant in value_variants(display_name):
            target = option_lookup.get(variant, "")
            if target:
                break
        if not target:
            continue
        for alias in aliases | {display_name}:
            for variant in value_variants(alias):
                alias_targets.setdefault(variant, set()).add(target)
    alias_lookup = dict(option_lookup)
    for variant, targets in alias_targets.items():
        if len(targets) == 1:
            alias_lookup.setdefault(variant, next(iter(targets)))
    normalized_box = []
    for value in my_box:
        target = next(
            (alias_lookup.get(variant) for variant in value_variants(value) if alias_lookup.get(variant)),
            "",
        )
        if target and target not in normalized_box:
            normalized_box.append(target)
    my_box = normalized_box
    alias_map_sets = {}
    search_alias_map_sets = {}
    for display_name, aliases in alias_sets.items():
        target = next(
            (
                option_lookup.get(variant)
                for variant in value_variants(display_name)
                if option_lookup.get(variant)
            ),
            "",
        )
        if target:
            for alias in aliases | {display_name}:
                search_alias_map_sets.setdefault(target, set()).add(alias)
                variants = value_variants(alias)
                if any(
                    len(alias_targets.get(variant, set())) != 1
                    for variant in variants
                ):
                    continue
                alias_map_sets.setdefault(target, set()).add(alias)
    alias_map = {
        target: sorted(aliases)
        for target, aliases in alias_map_sets.items()
    }
    search_text_map = {
        name: " ".join(
            [name, slugify_name(name)]
            + list(search_alias_map_sets.get(name, []))
        ).lower()
        for name in gallery_options
    }
    rank_groups = [
        {
            "id": str(item.get("group_id")),
            "name": str(item.get("group_name") or item.get("group_id")),
        }
        for item in (luck_stats.data.get("rank_groups") or [])
        if isinstance(item, dict) and item.get("group_id") is not None
    ]
    if not rank_groups:
        rank_groups = [{"id": str(luck_stats.rank_group), "name": "当前数据"}]

    return {
        "shared_bundle": shared_bundle,
        "options": gallery_options,
        "model_options": model_options or list(options),
        "gallery_options": gallery_options,
        "my_box": sorted(set(my_box), key=normalize_name),
        "account_profile": account_profile,
        "image_lookup": {
            **build_image_lookup(image_map),
            **build_image_lookup(remote_image_map),
        },
        "attr_map": attr_map,
        "family_map": family_map,
        "alias_map": alias_map,
        "alias_lookup": alias_lookup,
        "search_text_map": search_text_map,
        "attributes": ["全部"] + sorted({str(value) for value in attr_map.values() if str(value).strip()}),
        "families": ["全部"] + sorted({str(value) for value in family_map.values() if str(value).strip()}),
        "luck_stats_ready": bool(luck_stats.ready),
        "luck_stats_source": luck_stats.source_label,
        "rank_groups": rank_groups,
        "default_rank_group": str(luck_stats.rank_group),
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
def load_shared_recommender(model_path, meta_path, vocab_path, device, rank_group, model_stamp=None):
    selected_device = None if device == "auto" else device
    return LegacyBPRecommender(
        device=selected_device,
        stats_rank_group=rank_group,
    )


def phase_label(my_picks, enemy_picks, current_side):
    if len(my_picks) >= 5 and len(enemy_picks) >= 5:
        return "Ban 阶段"
    if current_side == "me":
        return "我方选人"
    if current_side == "enemy":
        return "对方选人"
    return "等待输入"


def make_signature(
    my_picks,
    enemy_picks,
    first_pick_side,
    my_leader,
    enemy_leader,
    my_box=(),
    use_my_box=False,
    rank_group="",
    device="auto",
    max_pair_candidates=0,
    my_final_ban="",
    enemy_final_ban="",
):
    payload = {
        "my_picks": list(my_picks),
        "enemy_picks": list(enemy_picks),
        "first_pick_side": first_pick_side,
        "my_leader": my_leader,
        "enemy_leader": enemy_leader,
        "my_box": sorted(set(my_box or ()), key=normalize_name),
        "use_my_box": bool(use_my_box),
        "rank_group": str(rank_group),
        "device": str(device),
        "max_pair_candidates": int(max_pair_candidates),
        "my_final_ban": my_final_ban or "",
        "enemy_final_ban": enemy_final_ban or "",
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
        "my_final_ban": "",
        "enemy_final_ban": "",
        "my_box": list(assets["my_box"]),
        "use_my_box": True,
        "top_k": 6,
        "max_pair_candidates": 400,
        "show_details": False,
        "device": "auto",
        "rank_group": assets.get("default_rank_group", "102"),
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

    previous_first_pick_side = st.session_state.setdefault(
        "_bp_first_pick_side", st.session_state.first_pick_side
    )
    if previous_first_pick_side != st.session_state.first_pick_side:
        st.session_state.my_picks, st.session_state.enemy_picks = legal_pick_prefix(
            st.session_state.my_picks,
            st.session_state.enemy_picks,
            st.session_state.first_pick_side,
        )
        clear_results()
        st.session_state._bp_first_pick_side = st.session_state.first_pick_side

    gallery_options = set(assets.get("gallery_options") or assets["options"])
    st.session_state.my_picks = clean_picker_values(
        st.session_state.my_picks,
        gallery_options,
        assets,
    )
    st.session_state.enemy_picks = clean_picker_values(
        st.session_state.enemy_picks,
        gallery_options,
        assets,
    )
    st.session_state.my_picks, st.session_state.enemy_picks = legal_pick_prefix(
        st.session_state.my_picks,
        st.session_state.enemy_picks,
        st.session_state.first_pick_side,
    )
    st.session_state.my_box = clean_picker_values(
        st.session_state.my_box,
        gallery_options,
        assets,
        max_items=None,
    )
    valid_rank_groups = {item["id"] for item in assets.get("rank_groups", [])}
    if st.session_state.rank_group not in valid_rank_groups:
        st.session_state.rank_group = assets.get("default_rank_group", "102")
    if st.session_state.my_leader not in st.session_state.my_picks:
        st.session_state.my_leader = ""
    if st.session_state.enemy_leader not in st.session_state.enemy_picks:
        st.session_state.enemy_leader = ""


def clear_results():
    st.session_state.pick_results = None
    st.session_state.enemy_results = None
    st.session_state.ban_results = None


def legal_pick_prefix(my_picks, enemy_picks, first_pick_side):
    """Keep the longest chronological prefix representable by side lists."""
    best = ([], [])
    for my_count in range(len(my_picks) + 1):
        for enemy_count in range(len(enemy_picks) + 1):
            candidate = (list(my_picks[:my_count]), list(enemy_picks[:enemy_count]))
            try:
                timeline_from_picks(*candidate, first_pick_side)
            except (TypeError, ValueError):
                continue
            if sum(map(len, candidate)) > sum(map(len, best)):
                best = candidate
    return best


def apply_recommendation_plan(result_type, monsters, assets):
    if result_type == "ban":
        if len(st.session_state.my_picks) != 5 or len(st.session_state.enemy_picks) != 5:
            return 0
        if st.session_state.my_final_ban or st.session_state.enemy_final_ban:
            return 0
        resolved = []
        for raw_monster in monsters:
            monster = resolve_option_name(raw_monster, assets)
            if monster and monster in set(assets.get("gallery_options") or assets.get("options") or []) and monster not in resolved:
                resolved.append(monster)
        if len(resolved) != 1:
            return 0
        if resolved[0] not in st.session_state.enemy_picks:
            return 0
        st.session_state.my_final_ban = resolved[0]
        clear_results()
        return 1
    if result_type not in {"pick", "enemy"}:
        return 0

    target_side = "me" if result_type == "pick" else "enemy"
    my_picks = st.session_state.my_picks
    enemy_picks = st.session_state.enemy_picks
    turn = len(my_picks) + len(enemy_picks) + 1
    expected_side = side_for_order(turn, st.session_state.first_pick_side)
    expected_count = infer_pick_count_for_side(
        my_picks,
        enemy_picks,
        st.session_state.first_pick_side,
        target_side,
    )
    if expected_side != target_side or expected_count not in {1, 2}:
        return 0

    target = my_picks if target_side == "me" else enemy_picks
    options = set(assets.get("gallery_options") or assets.get("options") or [])
    occupied = set(my_picks) | set(enemy_picks)
    resolved = []
    for raw_monster in monsters:
        monster = resolve_option_name(raw_monster, assets)
        if monster and monster in options and monster not in resolved:
            resolved.append(monster)
    if (
        len(resolved) != expected_count
        or len(target) + expected_count > 5
        or any(monster in occupied for monster in resolved)
    ):
        return 0

    target.extend(resolved)
    clear_results()
    return len(resolved)


def save_my_box():
    MY_BOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MY_BOX_PATH, "w", encoding="utf-8") as file:
        json.dump(sorted(set(st.session_state.my_box), key=normalize_name), file, ensure_ascii=False, indent=2)
    load_web_assets.clear()


def get_shared_model(assets):
    bundle = assets["shared_bundle"]
    if not bundle:
        raise FileNotFoundError(
            "没有找到旧版 BP 模型，请先运行 python -m training.all_models"
        )
    return load_shared_recommender(
        str(bundle["model_path"]),
        str(bundle["meta_path"]),
        str(bundle["vocab_path"]),
        st.session_state.device,
        st.session_state.rank_group,
        asset_cache_stamp(),
    )


def calculate_my_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    turn = len(my_picks) + len(enemy_picks) + 1
    first_pick_side = st.session_state.first_pick_side
    pick_count = infer_pick_count(my_picks, enemy_picks, first_pick_side)
    events = timeline_from_picks(my_picks, enemy_picks, first_pick_side).to_dict()
    state = DraftState(
        my_picks=tuple(my_picks),
        enemy_picks=tuple(enemy_picks),
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
        first_pick_side=first_pick_side,
        turn=turn,
        events=events,
    ).to_dict()
    shared_model = get_shared_model(assets)
    top_k = int(st.session_state.top_k)
    search_top_k = max(top_k * 2, top_k)
    plans = shared_model.recommend_pick_plans(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        pick_count=pick_count,
        turn=turn,
        first_pick_side=first_pick_side,
        candidate_pool=assets.get("gallery_options") or assets["options"],
        my_box=st.session_state.my_box,
        use_my_box_for_reco=bool(st.session_state.use_my_box),
        top_k=search_top_k,
        max_pair_candidates=int(st.session_state.max_pair_candidates),
        events=state["events"],
    )
    state.update(
        {
            "my_box": list(st.session_state.my_box),
            "use_my_box": bool(st.session_state.use_my_box),
            "max_pair_candidates": int(st.session_state.max_pair_candidates),
            "current_side": side_for_order(turn, first_pick_side),
        }
    )
    return BPDecisionEngine(model=shared_model, assets=assets).rank_my_plans(
        plans, state, top_k
    )


def calculate_enemy_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    turn = len(my_picks) + len(enemy_picks) + 1
    first_pick_side = st.session_state.first_pick_side
    pick_count = infer_pick_count_for_side(
        my_picks,
        enemy_picks,
        first_pick_side,
        "enemy",
    )
    events = timeline_from_picks(my_picks, enemy_picks, first_pick_side).to_dict()
    state = DraftState(
        my_picks=tuple(my_picks),
        enemy_picks=tuple(enemy_picks),
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
        first_pick_side=first_pick_side,
        turn=turn,
        events=events,
    ).to_dict()
    shared_model = get_shared_model(assets)
    top_k = int(st.session_state.top_k)
    search_top_k = max(top_k * 2, top_k)
    predictions = shared_model.recommend_enemy_next_picks(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        first_pick_side=first_pick_side,
        pick_count=pick_count,
        top_k=search_top_k,
        candidate_pool=assets.get("gallery_options") or assets["options"],
        max_pair_candidates=int(st.session_state.max_pair_candidates),
        turn=turn,
        events=state["events"],
    )
    state.update(
        {
            "my_box": list(st.session_state.my_box),
            "use_my_box": bool(st.session_state.use_my_box),
            "max_pair_candidates": int(st.session_state.max_pair_candidates),
            "current_side": side_for_order(turn, first_pick_side),
        }
    )
    return BPDecisionEngine(model=shared_model, assets=assets).rank_enemy_plans(
        predictions, state, top_k
    )


def calculate_ban_recommendations(assets):
    my_picks = list(st.session_state.my_picks)
    enemy_picks = list(st.session_state.enemy_picks)
    top_k = int(st.session_state.top_k)
    events = timeline_from_picks(
        my_picks,
        enemy_picks,
        st.session_state.first_pick_side,
    ).to_dict()
    state = DraftState(
        my_picks=tuple(my_picks),
        enemy_picks=tuple(enemy_picks),
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
        first_pick_side=st.session_state.first_pick_side,
        turn=11,
        events=events,
    ).to_dict()
    shared_model = get_shared_model(assets)
    return shared_model.recommend_ban_plans(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        top_k=top_k,
        first_pick_side=st.session_state.first_pick_side,
        my_leader=st.session_state.my_leader,
        enemy_leader=st.session_state.enemy_leader,
        enemy_ban=st.session_state.enemy_final_ban or None,
        events=state["events"],
    )


def filter_gallery_monsters(assets, search_text, attr_filter, family_filter, only_owned, only_with_image):
    monsters = list(assets["options"])
    if only_owned:
        owned = set(st.session_state.my_box)
        monsters = [monster for monster in monsters if monster in owned]

    search_text = str(search_text or "").strip()
    if search_text:
        monsters = [
            monster
            for monster in monsters
            if matches_monster_search(
                search_text,
                assets["search_text_map"].get(monster, monster),
            )
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
    model_option_keys = {
        normalize_name(value)
        for value in (assets.get("gallery_options") or assets.get("options") or [])
    }
    for monster in state_names:
        image_path = get_image_path(monster, assets["image_lookup"])
        state_items[monster] = {
            "name": monster,
            "attr": assets["attr_map"].get(monster, ""),
            "family": assets["family_map"].get(monster, ""),
            "image": load_image_data_uri(str(image_path)) if image_path else "",
            "enabled": normalize_name(monster) in model_option_keys,
        }

    return [
        state_items[monster]
        for monster in monsters
        if monster in state_items
    ], state_items


def clean_picker_values(values, options, assets=None, max_items=5):
    cleaned = []
    seen = set()
    for value in values or []:
        monster = resolve_option_name(value, assets or {"options": list(options)})
        if monster in options and monster not in seen:
            cleaned.append(monster)
            seen.add(monster)
        if max_items is not None and len(cleaned) >= max_items:
            break
    return cleaned


def apply_picker_value(value, assets):
    if not isinstance(value, dict):
        return False
    event_id = str(value.get("event_id", "") or "")
    if not event_id or event_id == st.session_state.picker_event_id:
        return False

    options = set(assets.get("gallery_options") or assets["options"])
    my_picks = clean_picker_values(value.get("my_picks"), options, assets)
    enemy_picks = clean_picker_values(value.get("enemy_picks"), options, assets)
    my_picks, enemy_picks = legal_pick_prefix(
        my_picks,
        enemy_picks,
        st.session_state.first_pick_side,
    )
    my_set = set(my_picks)
    enemy_picks = [monster for monster in enemy_picks if monster not in my_set]

    st.session_state.my_picks = my_picks
    st.session_state.enemy_picks = enemy_picks
    my_leader = resolve_option_name(value.get("my_leader"), assets)
    enemy_leader = resolve_option_name(value.get("enemy_leader"), assets)
    st.session_state.my_leader = (
        my_leader
        if my_leader in my_picks
        else ""
    )
    st.session_state.enemy_leader = (
        enemy_leader
        if enemy_leader in enemy_picks
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
            rerun_fragment_or_app()
    with page_columns[1]:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem'>第 {st.session_state.gallery_page}/{total_pages} 页 · 共 {len(gallery_monsters)} 个</div>",
            unsafe_allow_html=True,
        )
    with page_columns[2]:
        if st.button("下一页", key="gallery_next", use_container_width=True):
            st.session_state.gallery_page = min(total_pages, st.session_state.gallery_page + 1)
            rerun_fragment_or_app()

    items, state_items = build_picker_payload(page_items, assets)
    current_turn = len(st.session_state.my_picks) + len(st.session_state.enemy_picks) + 1
    current_side = side_for_order(current_turn, st.session_state.first_pick_side) or "none"
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
        rerun_fragment_or_app()

    current_turn = len(st.session_state.my_picks) + len(st.session_state.enemy_picks) + 1
    current_side = side_for_order(current_turn, st.session_state.first_pick_side)
    if st.session_state.my_final_ban or st.session_state.enemy_final_ban:
        st.info(
            "；".join(
                value for value in (
                    f"我方 Ban：{st.session_state.my_final_ban}" if st.session_state.my_final_ban else "",
                    f"对方 Ban：{st.session_state.enemy_final_ban}" if st.session_state.enemy_final_ban else "",
                )
                if value
            )
        )
    signature = make_signature(
        st.session_state.my_picks,
        st.session_state.enemy_picks,
        st.session_state.first_pick_side,
        st.session_state.my_leader,
        st.session_state.enemy_leader,
        st.session_state.my_box,
        st.session_state.use_my_box,
        st.session_state.rank_group,
        st.session_state.device,
        st.session_state.max_pair_candidates,
        st.session_state.my_final_ban,
        st.session_state.enemy_final_ban,
    )

    action_columns = st.columns(4)
    with action_columns[0]:
        if st.button("清空 BP 状态", key="clear_bp_fragment", use_container_width=True):
            st.session_state.my_picks = []
            st.session_state.enemy_picks = []
            st.session_state.my_leader = ""
            st.session_state.enemy_leader = ""
            st.session_state.my_final_ban = ""
            st.session_state.enemy_final_ban = ""
            clear_results()
            rerun_fragment_or_app()
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
            except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
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
            except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
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
            except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
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
    clear_results()


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
                    rerun_fragment_or_app()


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
    search_text = str(st.session_state.box_search or "").strip()
    if search_text:
        monsters = [
            monster
            for monster in monsters
            if matches_monster_search(
                search_text,
                assets["search_text_map"].get(monster, monster),
            )
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
            rerun_fragment_or_app()
    with box_nav_label:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem'>第 {st.session_state.box_page} 页</div>",
            unsafe_allow_html=True,
        )
    with box_nav_right:
        if st.button("下一页", key="box_next", use_container_width=True):
            st.session_state.box_page = min(total_pages, st.session_state.box_page + 1)
            rerun_fragment_or_app()

    save_column, clear_column = st.columns(2)
    with save_column:
        if st.button("保存 Box", key="save_box", use_container_width=True):
            save_my_box()
            st.success("已保存")
    with clear_column:
        if st.button("清空 Box", key="clear_box", use_container_width=True):
            st.session_state.my_box = []
            rerun_fragment_or_app()


def render_result_card(index, item, assets, result_type):
    monsters = [value for value in item["plan"] if value]
    score = item["plan_score"]

    detail = item.get("detail", {}) or {}
    win_rate = item.get("estimated_win_rate")
    primary_label = "估计胜率"
    if result_type == "pick" and detail.get("adversarial_win_rate") is not None:
        win_rate = detail.get("adversarial_win_rate")
        primary_label = "对手反制后胜率"
    elif result_type == "enemy" and detail.get("best_response_win_rate") is not None:
        win_rate = detail.get("best_response_win_rate")
        primary_label = "我方应对后胜率"
    elif result_type == "ban" and detail.get("post_ban_win_rate") is not None:
        win_rate = detail.get("post_ban_win_rate")
        primary_label = "Ban 后胜率"
    if win_rate is not None and float(win_rate or 0.0) > 0.0:
        win_rate_text = f"{float(win_rate):.2%}"
    else:
        win_rate_text = "未启用"

    if result_type == "ban" and "pre_ban_win_rate" in detail:
        secondary_label = "Ban 前"
        secondary_value = f"{float(detail['pre_ban_win_rate']):.2%}"
    elif "win_rate_gain" in detail:
        secondary_label = "变化"
        secondary_value = f"{float(detail['win_rate_gain']):+.2%}"
    elif "post_enemy_pick_win_rate" in detail:
        secondary_label = "我方胜率"
        secondary_value = f"{float(detail['post_enemy_pick_win_rate']):.2%}"
    elif "post_ban_win_rate" in detail:
        secondary_label = "Ban 后"
        secondary_value = f"{float(detail['post_ban_win_rate']):.2%}"
    else:
        secondary_label = "动作"
        secondary_value = str(detail.get("action_type") or item.get("source", "shared_policy"))

    ranking_source = detail.get("ranking_source") or item.get("source") or "shared_policy"
    source_text = {
        "shared_policy": "Legacy BP policy",
        "relation": "Relation + 统计证据",
        "relation_ban": "Relation Ban",
        "bp_draft_value_model": "Full Draft Value + Ban",
        "robust_search_value_3ply_shared_stats_mechanism": "三层鲁棒搜索 + Value + 机制",
        "robust_enemy_search_value_shared_stats_mechanism": "对手搜索 + Value + 机制",
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
                <div><span>{html.escape(primary_label)}</span><b>{win_rate_text}</b></div>
                <div><span>{html.escape(secondary_label)}</span><b>{html.escape(secondary_value)}</b></div>
                <div><span>来源</span><b>{html.escape(source_text)}</b></div>
            </div>
            {reason_markup}
        </div>
    """
    st.markdown(card_markup, unsafe_allow_html=True)
    if result_type in {"pick", "enemy", "ban"}:
        if st.button(
            "采用",
            key=f"recommendation_apply_{result_type}_{index}",
            type="primary",
            use_container_width=True,
        ):
            added = apply_recommendation_plan(result_type, monsters, assets)
            if added:
                rerun_fragment_or_app()
            else:
                st.warning("当前推荐与 BP 轮次不匹配，或无法解析完整动作。")
    if st.session_state.show_details and detail:
        with st.expander("模型明细", expanded=False):
            st.json(detail)


def render_results(title, results, assets, result_type):
    st.subheader(title)
    if not results:
        st.info("没有可显示的推荐结果。")
        return
    def display_rate(item):
        detail = item.get("detail", {}) or {}
        if result_type == "ban":
            value = detail.get("post_ban_win_rate", item.get("estimated_win_rate"))
        elif result_type == "enemy":
            value = detail.get("best_response_win_rate", item.get("estimated_win_rate"))
        else:
            value = detail.get("adversarial_win_rate", item.get("estimated_win_rate"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0 if result_type != "enemy" else 1.0

    if result_type == "enemy":
        ordered_results = sorted(
            list(results),
            key=lambda item: (
                display_rate(item),
                -float(item.get("plan_score", 0.0)),
            ),
        )
    else:
        ordered_results = sorted(
            list(results),
            key=lambda item: (
                display_rate(item),
                float(item.get("plan_score", 0.0)),
            ),
            reverse=True,
        )
    for start in range(0, len(ordered_results), 3):
        result_row = ordered_results[start : start + 3]
        columns = st.columns(3)
        for column, (index, item) in zip(
            columns,
            enumerate(result_row, start=start + 1),
        ):
            with column:
                render_result_card(index, item, assets, result_type)


def render_model_status(assets):
    bundle = assets["shared_bundle"]
    if bundle:
        st.success(f"模型：{bundle['label']}")
    else:
        st.error("旧版 BP 模型：未找到，请先运行 python -m training.all_models")
    account_profile = assets.get("account_profile")
    if account_profile is None:
        st.caption("账号档案：未加载，推荐不限制我方 Box")
    else:
        st.caption("账号档案：已加载，仅用于识别我方拥有的魔灵")


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

    assets = load_web_assets(asset_cache_stamp())
    initialize_state(assets)
    sample_count = load_sample_count(str(SAMPLES_DB_PATH))

    with st.sidebar:
        st.header("控制")
        st.checkbox("只从我的 Box 推荐", key="use_my_box")
        rank_group_map = {
            item["id"]: item["name"] for item in assets.get("rank_groups", [])
        }
        st.selectbox(
            "统计段位",
            options=list(rank_group_map),
            format_func=lambda value: rank_group_map[value],
            key="rank_group",
            help="统计层使用所选段位；模型仍会结合当前阵容和机制约束。",
        )
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
