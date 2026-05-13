from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import Counter
from itertools import combinations
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageGrab, ImageOps, ImageTk

from user_monster_profile import (
    ACCOUNT_PROFILE_DEFAULT,
    BUILD_STATE_LABELS,
    GRADE_OPTIONS,
    ROLE_TAG_OPTIONS,
    make_default_monster_build,
    normalize_account_profile,
    normalize_monster_build_entry,
    normalize_monster_builds,
)


ROOT_DIR = Path(__file__).resolve().parent
MY_BOX_PATH = ROOT_DIR / "json" / "my_box.json"
ACCOUNT_PROFILE_PATH = ROOT_DIR / "json" / "account_profile.json"
MONSTER_BUILDS_PATH = ROOT_DIR / "json" / "monster_builds.json"
IMAGE_MAP_PATH = ROOT_DIR / "json" / "monster_images.json"
ATTR_MAP_PATH = ROOT_DIR / "json" / "monster_attributes.json"
FAMILY_MAP_PATH = ROOT_DIR / "json" / "monster_families.json"
RELATION_META_PATH = ROOT_DIR / "models" / "bp_relation_meta.json"
RELATION_VOCAB_PATH = ROOT_DIR / "models" / "bp_relation_vocab.json"
SW_UI_DIR = ROOT_DIR / "sw_ui_detection"
SW_UI_OUTPUT_DIR = SW_UI_DIR / "output"
APP_BP_IMPORT_PATH = SW_UI_OUTPUT_DIR / "app_bp_import_latest.png"
APP_BP_JSON_PATH = SW_UI_OUTPUT_DIR / "app_bp_capture_latest.json"

ACCOUNT_PROFILE_IMPORT_ALIASES = {
    "profile_name": "profile_name",
    "name": "profile_name",
    "账号名称": "profile_name",
    "账号名": "profile_name",
    "账号名称": "profile_name",
    "账号名": "profile_name",
    "contest_speed_1": "contest_speed_1",
    "抢速一速": "contest_speed_1",
    "一速": "contest_speed_1",
    "抢速一速": "contest_speed_1",
    "一速": "contest_speed_1",
    "contest_speed_2": "contest_speed_2",
    "抢速二速": "contest_speed_2",
    "二速": "contest_speed_2",
    "抢速二速": "contest_speed_2",
    "二速": "contest_speed_2",
    "fast_strip_speed": "fast_strip_speed",
    "高速剥条": "fast_strip_speed",
    "剥条速度": "fast_strip_speed",
    "高速剥条": "fast_strip_speed",
    "剥条速度": "fast_strip_speed",
    "followup_cc_speed": "followup_cc_speed",
    "跟进控制速度": "followup_cc_speed",
    "控制跟速": "followup_cc_speed",
    "跟进控制速度": "followup_cc_speed",
    "控制跟速": "followup_cc_speed",
    "followup_dps_speed": "followup_dps_speed",
    "跟进输出速度": "followup_dps_speed",
    "输出跟速": "followup_dps_speed",
    "跟进输出速度": "followup_dps_speed",
    "输出跟速": "followup_dps_speed",
    "turn2_support_speed": "turn2_support_speed",
    "后手辅助速度": "turn2_support_speed",
    "turn2速度": "turn2_support_speed",
    "后手辅助速度": "turn2_support_speed",
    "turn2速度": "turn2_support_speed",
    "speed_confidence": "speed_confidence",
    "速度信心": "speed_confidence",
    "速度信心": "speed_confidence",
    "box_depth_speed": "box_depth_speed",
    "先手深度": "box_depth_speed",
    "先手深度": "box_depth_speed",
    "box_depth_bruiser": "box_depth_bruiser",
    "后手肉队深度": "box_depth_bruiser",
    "肉队深度": "box_depth_bruiser",
    "后手肉队深度": "box_depth_bruiser",
    "肉队深度": "box_depth_bruiser",
    "box_depth_anti_cc": "box_depth_anti_cc",
    "反控深度": "box_depth_anti_cc",
    "anti_cc深度": "box_depth_anti_cc",
    "反控深度": "box_depth_anti_cc",
    "anti_cc深度": "box_depth_anti_cc",
    "box_depth_tech": "box_depth_tech",
    "对策卡深度": "box_depth_tech",
    "tech深度": "box_depth_tech",
    "对策卡深度": "box_depth_tech",
    "tech深度": "box_depth_tech",
    "notes": "notes",
    "备注": "notes",
    "备注": "notes",
}

BP_GALLERY_IMAGE_SIZE = (74, 74)
BP_GALLERY_COLUMNS = 13
BP_GALLERY_PAGE_ROWS = 4
BOX_GALLERY_IMAGE_SIZE = (76, 76)
BOX_GALLERY_COLUMNS = 11
CURRENT_BOX_COLUMNS = 4
SIDEBAR_PLAN_IMAGE_SIZE = (40, 40)
SIDEBAR_RESULT_IMAGE_SIZE = (48, 48)
STATE_SLOT_IMAGE_SIZE = (58, 58)
STATE_SLOT_CARD_SIZE = {"width": 50, "height": 130}
STATE_INLINE_IMAGE_SIZE = (64, 64)


def bootstrap_tk_env():
    prefix = Path(sys.prefix)
    tcl_candidates = [
        prefix / "Library" / "lib" / "tcl8.6",
        prefix / "tcl" / "tcl8.6",
    ]
    tk_candidates = [
        prefix / "Library" / "lib" / "tk8.6",
        prefix / "tcl" / "tk8.6",
    ]

    if not os.environ.get("TCL_LIBRARY"):
        for candidate in tcl_candidates:
            if (candidate / "init.tcl").exists():
                os.environ["TCL_LIBRARY"] = str(candidate)
                break

    if not os.environ.get("TK_LIBRARY"):
        for candidate in tk_candidates:
            if (candidate / "tk.tcl").exists():
                os.environ["TK_LIBRARY"] = str(candidate)
                break


def load_json_file(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_file(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def normalize_name(name):
    if name is None:
        return ""
    return str(name).strip().lower()


def slugify_fallback(name):
    text = normalize_name(name)
    text = text.replace(" - ", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    return "-".join(text.split())


def resolve_image_path(path_value):
    if not path_value:
        return None

    raw = str(path_value).strip().replace("/", "\\")
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(ROOT_DIR / path)
        candidates.append(ROOT_DIR / "images" / path.name)
        candidates.append(ROOT_DIR / "images" / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_supported_image_file(path_value):
    try:
        suffix = Path(path_value).suffix.lower()
    except Exception:
        return False
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_image_lookup(image_map):
    lookup = {}
    for raw_key, raw_path in (image_map or {}).items():
        resolved = resolve_image_path(raw_path)
        if not resolved:
            continue

        key_text = str(raw_key).strip()
        variants = {
            key_text,
            key_text.lower(),
            key_text.replace("-", " "),
            key_text.replace(" ", "-"),
            key_text.lower().replace("-", " "),
            key_text.lower().replace(" ", "-"),
        }
        for variant in variants:
            lookup[normalize_name(variant)] = resolved
    return lookup


def get_image_path_for_monster(monster, image_lookup):
    if not monster:
        return None

    candidates = [
        str(monster).strip(),
        str(monster).strip().lower(),
        str(monster).strip().replace("-", " "),
        str(monster).strip().replace(" ", "-"),
        slugify_fallback(monster),
    ]
    for candidate in candidates:
        path = image_lookup.get(normalize_name(candidate))
        if path:
            return path
    return None


def build_pick_frequency(samples):
    freq = Counter()
    for sample in samples or []:
        for monster in sample.get("my_picks", []):
            if monster:
                freq[monster] += 1
        for monster in sample.get("enemy_picks", []):
            if monster:
                freq[monster] += 1
        next_pick = sample.get("next_pick")
        if next_pick:
            freq[next_pick] += 1
    return freq


def load_neural_monster_metadata():
    meta = load_json_file(RELATION_META_PATH, {})
    vocab = load_json_file(RELATION_VOCAB_PATH, {})
    display_name_map = meta.get("display_name_map", {}) or {}
    monster_frequency = meta.get("monster_frequency", {}) or {}

    blocked = {"[PAD]", "[UNK]", "", None}
    if isinstance(vocab, dict):
        slug_items = sorted(vocab.items(), key=lambda item: int(item[1]))
    else:
        slug_items = []

    monster_pool = []
    pick_freq = Counter()
    seen = set()
    for slug, _idx in slug_items:
        if slug in blocked:
            continue
        display_name = str(display_name_map.get(slug) or slug).strip()
        if not display_name or display_name in seen:
            continue
        seen.add(display_name)
        monster_pool.append(display_name)
        try:
            pick_freq[display_name] = int(float(monster_frequency.get(slug, monster_frequency.get(str(_idx), 0)) or 0))
        except Exception:
            pick_freq[display_name] = 0

    return monster_pool, pick_freq


def sort_monsters_by_pick_frequency(monsters, pick_freq):
    return sorted(monsters, key=lambda m: (-pick_freq.get(m, 0), m))


def infer_bp_stage(my_picks, enemy_picks):
    my_count = len(my_picks or [])
    enemy_count = len(enemy_picks or [])

    if my_count >= 5 and enemy_count >= 5:
        return "ban"

    if my_count < 5:
        remain = 5 - my_count
        if remain == 1 or (my_count == 0 and enemy_count == 0):
            return "pick_single"
        return "pick_multi"

    return "idle"


def infer_turn_from_state(my_picks, enemy_picks, first_pick_side):
    total = len(my_picks) + len(enemy_picks)
    turn = total + 1

    if first_pick_side == "me":
        my_turns = {1, 4, 5, 8, 9}
    else:
        my_turns = {2, 3, 6, 7, 10}

    if turn not in my_turns:
        return None
    return turn


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, horizontal=False, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview) if horizontal else None

        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        if self.h_scroll:
            self.canvas.configure(xscrollcommand=self.h_scroll.set)
            self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_frame_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _is_descendant_widget(self, widget):
        current = widget
        while current is not None:
            if current is self or current is self.canvas or current is self.inner:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_mousewheel(self, event):
        hovered = self.winfo_containing(event.x_root, event.y_root)
        if not self._is_descendant_widget(hovered):
            return

        if getattr(event, "num", None) == 4:
            delta_units = -1
        elif getattr(event, "num", None) == 5:
            delta_units = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            delta_units = int(-delta / 120)
            if delta_units == 0:
                delta_units = -1 if delta > 0 else 1

        self.canvas.yview_scroll(delta_units, "units")

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()


class DesktopBPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("魔灵召唤 BP 桌面版")
        self.root.geometry("1800x1050")
        self.root.minsize(1500, 900)

        self.samples = []
        self.recommender = None
        self.relation_recommender = None
        self.comp_recommender = None
        self.ban_value_recommender = None
        self.bp_analyzer = None
        self._core_data_ready = False
        self._profile_tab_loaded = False
        self._box_tab_loaded = False
        self._lazy_bootstrap_started = False

        self.monster_pool = []
        self.image_map = {}
        self.attr_map = {}
        self.family_map = {}
        self.image_lookup = {}
        self.pick_freq = Counter()

        self.photo_cache = {}
        self.my_box_state = sorted(set(load_json_file(MY_BOX_PATH, [])))
        self.account_profile_state = normalize_account_profile(load_json_file(ACCOUNT_PROFILE_PATH, {}))
        self.monster_builds_state = normalize_monster_builds(load_json_file(MONSTER_BUILDS_PATH, {}))
        self.my_picks = []
        self.enemy_picks = []
        self.banned_monsters = []
        self.my_final_ban = None
        self.enemy_final_ban = None
        self.pick_plans = []
        self.enemy_predictions = []
        self.ban_plans = []
        self._active_task_id = 0
        self._active_task_name = None
        self._active_task_status_job = None
        self._pending_auto_recommend = False
        self._task_result_queue = queue.Queue()
        self.bp_recognition_summary_var = tk.StringVar(value="No BP recognition result yet")
        self.bp_recognition_result = None
        self.bp_recognition_source = None
        self.bp_recognition_image_path = None
        self.bp_enemy_candidate_top_n_var = tk.IntVar(value=4)

        self.status_var = tk.StringVar(value="就绪")
        self.phase_var = tk.StringVar(value="等待操作")
        self.last_action_var = tk.StringVar(value="")

        self.first_pick_var = tk.StringVar(value="me")
        self.search_var = tk.StringVar(value="")
        self.only_owned_in_picker_var = tk.BooleanVar(value=False)
        self.use_my_box_for_reco_var = tk.BooleanVar(value=True)
        self.only_with_image_var = tk.BooleanVar(value=False)
        self.attr_filter_var = tk.StringVar(value="全部")
        self.family_filter_var = tk.StringVar(value="全部")
        self.top_k_var = tk.IntVar(value=6)
        self.min_similarity_var = tk.DoubleVar(value=3.0)
        self.min_support_var = tk.IntVar(value=1)
        self.candidate_pool_size_var = tk.IntVar(value=10)
        self.bp_page_size_var = tk.IntVar(value=BP_GALLERY_COLUMNS * BP_GALLERY_PAGE_ROWS)
        self.box_page_size_var = tk.IntVar(value=50)
        self.my_leader_var = tk.StringVar(value="")
        self.enemy_leader_var = tk.StringVar(value="")
        self.enemy_ban_candidate_var = tk.StringVar(value="")
        self.profile_search_var = tk.StringVar(value="")
        self.profile_attr_filter_var = tk.StringVar(value="鍏ㄩ儴")
        self.profile_family_filter_var = tk.StringVar(value="鍏ㄩ儴")
        self.profile_only_box_var = tk.BooleanVar(value=False)
        self.profile_only_saved_var = tk.BooleanVar(value=False)
        self.profile_only_with_image_var = tk.BooleanVar(value=True)
        self.profile_page_size_var = tk.IntVar(value=52)
        self.comp_reco_mode_var = tk.StringVar(value="当前赛季强势三系阵容")
        self.comp_reco_top_k_var = tk.IntVar(value=10)
        self.comp_box_size_var = tk.IntVar(value=10)
        self.comp_reco_use_box_var = tk.BooleanVar(value=False)
        self.comp_core_search_var = tk.StringVar(value="")
        self.comp_reco_season_var = tk.StringVar(value="全部样本")
        self.comp_core_monsters = []
        self.comp_recommendations = []

        self.account_profile_name_var = tk.StringVar(value="")
        self.account_contest_speed_1_var = tk.StringVar(value="")
        self.account_contest_speed_2_var = tk.StringVar(value="")
        self.account_fast_strip_speed_var = tk.StringVar(value="")
        self.account_followup_cc_speed_var = tk.StringVar(value="")
        self.account_followup_dps_speed_var = tk.StringVar(value="")
        self.account_turn2_support_speed_var = tk.StringVar(value="")
        self.account_speed_confidence_var = tk.StringVar(value="B")
        self.account_box_depth_speed_var = tk.StringVar(value="B")
        self.account_box_depth_bruiser_var = tk.StringVar(value="B")
        self.account_box_depth_anti_cc_var = tk.StringVar(value="B")
        self.account_box_depth_tech_var = tk.StringVar(value="B")
        self.account_notes_var = tk.StringVar(value="")

        self.profile_selected_monster = None
        self.monster_build_state_var = tk.IntVar(value=0)
        self.monster_actual_speed_var = tk.StringVar(value="")
        self.monster_rune_tier_var = tk.StringVar(value="C")
        self.monster_artifact_tier_var = tk.StringVar(value="C")
        self.monster_skillup_complete_var = tk.BooleanVar(value=False)
        self.monster_confidence_score_var = tk.StringVar(value="0.0")
        self.monster_preferred_build_var = tk.StringVar(value="")
        self.monster_role_tags_var = tk.StringVar(value="")
        self.monster_notes_var = tk.StringVar(value="")
        self.box_search_var = tk.StringVar(value="")
        self.box_attr_filter_var = tk.StringVar(value="全部")
        self.box_only_owned_var = tk.BooleanVar(value=False)
        self.box_only_with_image_var = tk.BooleanVar(value=True)
        self.profile_gallery_page = 1

        self.bp_gallery_page = 1
        self.box_gallery_page = 1
        self.attributes = ["全部"]
        self.families = ["全部"]

        self._configure_style()
        self._build_ui()
        self._bind_var_events()
        self._load_profile_state_into_vars()
        self._render_initial_empty_state()
        self.root.after(50, self._start_lazy_bootstrap)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(bg="#e8edf6")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Sidebar.TFrame", background="#eef2f7")
        style.configure("Section.TLabelframe", background="#ffffff")
        style.configure("Section.TLabelframe.Label", font=("", 10, "bold"))

    def _render_initial_empty_state(self):
        self.set_phase()
        self.refresh_sidebar_pick_plans()
        self.refresh_bp_gallery()
        self.refresh_bp_state_panel()
        self.set_status("Core data is loading in background")

    def _start_lazy_bootstrap(self):
        if self._lazy_bootstrap_started:
            return
        self._lazy_bootstrap_started = True

        def worker():
            try:
                result = self._lazy_load_core_data()
            except Exception as exc:
                result = {"error": exc, "traceback": traceback.format_exc()}
            self.root.after(0, lambda: self._apply_lazy_bootstrap_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _lazy_load_core_data(self):
        image_map = load_json_file(IMAGE_MAP_PATH, {})
        attr_map = load_json_file(ATTR_MAP_PATH, {})
        family_map = load_json_file(FAMILY_MAP_PATH, {})
        image_lookup = build_image_lookup(image_map)
        monster_pool, pick_freq = load_neural_monster_metadata()
        if not monster_pool:
            fallback_names = set(image_map) | set(attr_map) | set(family_map)
            monster_pool = sorted(str(name).strip() for name in fallback_names if str(name).strip())
        return {
            "monster_pool": monster_pool,
            "pick_freq": pick_freq,
            "image_map": image_map,
            "attr_map": attr_map,
            "family_map": family_map,
            "image_lookup": image_lookup,
        }

    def _apply_lazy_bootstrap_result(self, result):
        if isinstance(result, dict) and result.get("error") is not None:
            print(result.get("traceback", ""))
            self.set_status(f"Core data load failed: {result['error']}")
            return

        self.samples = []
        self.recommender = None
        self.monster_pool = result["monster_pool"]
        self.pick_freq = result["pick_freq"]
        self.image_map = result["image_map"]
        self.attr_map = result["attr_map"]
        self.family_map = result["family_map"]
        self.image_lookup = result["image_lookup"]
        self.attributes = ["全部"] + sorted(set(v for v in self.attr_map.values() if v))
        self.families = ["全部"] + sorted(set(v for v in self.family_map.values() if v))
        self._refresh_filter_widget_values()
        self._core_data_ready = True
        self.refresh_all()
        self.set_status("Core data loaded")
        self.root.after(0, self._run_auto_recommendation_if_needed)

    def _refresh_filter_widget_values(self):
        for widget_name in ("bp_attr_filter_combo", "profile_attr_filter_combo", "box_attr_filter_combo"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(values=self.attributes)
        for widget_name in ("bp_family_filter_combo", "profile_family_filter_combo"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(values=self.families)

    def _ensure_core_data_ready(self, silent=False):
        if self._core_data_ready:
            return True
        if not silent:
            messagebox.showinfo("Notice", "Core data is still loading, please try again shortly.")
        return False

    def _current_tab_name(self):
        if not hasattr(self, "notebook"):
            return "bp"
        selected = self.notebook.select()
        if selected == str(self.profile_tab):
            return "profile"
        if selected == str(self.box_tab):
            return "box"
        return "bp"

    def _on_tab_changed(self, _event=None):
        tab_name = self._current_tab_name()
        if tab_name == "profile" and not self._profile_tab_loaded:
            self.root.after_idle(self.refresh_profile_tab)
        elif tab_name == "box" and not self._box_tab_loaded:
            self.root.after_idle(self.refresh_box_tab)

    def _build_ui(self):
        root_container = ttk.Frame(self.root)
        root_container.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(root_container, style="Sidebar.TFrame", padding=12, width=290)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(root_container, padding=(10, 10, 10, 0))
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sidebar_v3()
        self._build_main_tabs()

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_sidebar_v3(self):
        title = ttk.Label(self.sidebar, text="BP 操作", font=("", 13, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        action_frame = ttk.LabelFrame(self.sidebar, text="对局操作", padding=10, style="Section.TLabelframe")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(action_frame, text="刷新当前界面", command=self.refresh_all).pack(fill=tk.X, pady=3)
        ttk.Button(action_frame, text="清空全部", command=self.clear_all).pack(fill=tk.X, pady=3)

        self.sidebar_pick_plans_section = ttk.LabelFrame(self.sidebar, text="推荐结果", padding=10, style="Section.TLabelframe")
        self.sidebar_pick_plans_section.pack(fill=tk.BOTH, expand=True)
        self.sidebar_pick_plans_scroll = ScrollableFrame(self.sidebar_pick_plans_section)
        self.sidebar_pick_plans_scroll.pack(fill=tk.BOTH, expand=True)

    def refresh_sidebar_pick_plans(self):
        if not hasattr(self, "sidebar_pick_plans_scroll"):
            return

        self.sidebar_pick_plans_scroll.clear()
        parent = self.sidebar_pick_plans_scroll.inner

        if not self.pick_plans and not self.enemy_predictions and not self.ban_plans:
            ttk.Label(parent, text="No results yet", foreground="#6b7280").pack(anchor="w")
            return

        if self.pick_plans:
            ttk.Label(parent, text="Pick Plans", foreground="#3153a1", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))
            pick_grid = ttk.Frame(parent)
            pick_grid.pack(fill=tk.X, pady=(0, 8))
            columns = 2
            plan_limit = max(1, int(self.top_k_var.get()))
            for idx, plan in enumerate(self.pick_plans[:plan_limit], start=1):
                card = self.build_sidebar_plan_card(pick_grid, plan, idx)
                row = (idx - 1) // columns
                col = (idx - 1) % columns
                card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
                pick_grid.grid_columnconfigure(col, weight=1)

        if self.enemy_predictions:
            ttk.Label(parent, text="Enemy Next", foreground="#3153a1", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))
            pred_grid = ttk.Frame(parent)
            pred_grid.pack(fill=tk.X, pady=(0, 8))
            columns = 2
            plan_limit = max(1, int(self.top_k_var.get()))
            for idx, prediction in enumerate(self.enemy_predictions[:plan_limit], start=1):
                card = self.build_sidebar_prediction_card(pred_grid, prediction, idx)
                row = (idx - 1) // columns
                col = (idx - 1) % columns
                card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
                pred_grid.grid_columnconfigure(col, weight=1)

        if self.ban_plans:
            ttk.Label(parent, text="Ban Plans", foreground="#3153a1", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))
            ban_grid = ttk.Frame(parent)
            ban_grid.pack(fill=tk.X)
            columns = 2
            plan_limit = max(1, int(self.top_k_var.get()))
            for idx, plan in enumerate(self.ban_plans[:plan_limit], start=1):
                card = self.build_sidebar_ban_card(ban_grid, plan, idx)
                row = (idx - 1) // columns
                col = (idx - 1) % columns
                card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
                ban_grid.grid_columnconfigure(col, weight=1)

    def build_sidebar_plan_card(self, parent, plan, idx):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.RIDGE)
        tk.Label(card, text=f"Plan {idx}", bg="#ffffff", font=("", 9, "bold")).pack(anchor="w", padx=6, pady=(5, 3))

        images_row = tk.Frame(card, bg="#ffffff")
        images_row.pack(fill=tk.X, padx=6)
        for monster in plan.get("plan", []):
            block = tk.Frame(images_row, bg="#ffffff")
            block.pack(side=tk.LEFT, padx=2, pady=(0, 3))
            photo = self.get_photo(monster, size=SIDEBAR_PLAN_IMAGE_SIZE)
            label = tk.Label(block, image=photo if photo else "", text="" if photo else "No Img", bg="#ffffff")
            label.image = photo
            label.pack()
            short_name = monster if len(monster) <= 6 else f"{monster[:5]}..."
            tk.Label(
                block,
                text=short_name,
                bg="#ffffff",
                font=("", 7),
                wraplength=48,
                justify=tk.CENTER,
            ).pack()

        info = tk.Frame(card, bg="#ffffff")
        info.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(info, text=f"Score {float(plan.get('plan_score', 0.0)):.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)
        tk.Label(info, text=f"Win {float(plan.get('estimated_win_rate', 0.0)):.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)
        rerank_value = float(plan.get("rerank_value", plan.get("detail", {}).get("partial_value_win_rate", 0.0)))
        tk.Label(info, text=f"Rerank {rerank_value:.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)

        tk.Button(card, text="Apply", command=lambda p=plan: self.apply_pick_plan(p), font=("", 8, "bold"), height=1).pack(fill=tk.X, padx=6, pady=(0, 6))
        return card

    def build_sidebar_ban_card(self, parent, plan, idx):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.RIDGE)
        tk.Label(card, text=f"Ban {idx}", bg="#ffffff", font=("", 9, "bold")).pack(anchor="w", padx=6, pady=(5, 3))

        monster = plan.get("ban") or plan.get("monster", "")
        photo = self.get_photo(monster, size=SIDEBAR_RESULT_IMAGE_SIZE)
        image_label = tk.Label(card, image=photo if photo else "", text="" if photo else "No Img", bg="#ffffff")
        image_label.image = photo
        image_label.pack(pady=(0, 3))
        short_name = monster if len(monster) <= 8 else f"{monster[:7]}..."
        tk.Label(
            card,
            text=short_name,
            bg="#ffffff",
            font=("", 8, "bold"),
            wraplength=82,
            justify=tk.CENTER,
        ).pack(padx=6)

        detail = plan.get("detail", {}) if isinstance(plan.get("detail"), dict) else {}
        info = tk.Frame(card, bg="#ffffff")
        info.pack(fill=tk.X, padx=6, pady=(4, 4))
        tk.Label(info, text=f"Win {float(plan.get('estimated_win_rate', 0.0)):.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)
        tk.Label(info, text=f"Gain {float(detail.get('win_rate_gain', 0.0)):+.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)

        tk.Button(card, text="Apply Ban", command=lambda m=monster: self.apply_my_final_ban(m), font=("", 8, "bold"), height=1).pack(fill=tk.X, padx=6, pady=(0, 6))
        return card

    def _build_main_tabs(self):
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.bp_tab = ttk.Frame(self.notebook)
        self.profile_tab = ttk.Frame(self.notebook)
        self.box_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bp_tab, text="BP 模式")
        self.notebook.add(self.profile_tab, text="魔灵资料")
        self.notebook.add(self.box_tab, text="我的 Box 管理")

        self.notebook.hide(self.profile_tab)

        self._build_bp_tab()
        self._build_box_tab()
        self.notebook.tab(self.bp_tab, text="BP 模式")
        self.notebook.tab(self.profile_tab, text="阵容推荐")
        self.notebook.tab(self.box_tab, text="我的 Box 管理")

    def _build_bp_tab(self):
        layout = ttk.Frame(self.bp_tab)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=11)
        layout.columnconfigure(1, weight=3)
        layout.rowconfigure(0, weight=1)

        self.bp_left = ttk.Frame(layout)
        self.bp_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.bp_right = ttk.Frame(layout, width=360)
        self.bp_right.grid(row=0, column=1, sticky="nsew")

        self._build_bp_gallery_panel_v2()
        self._build_bp_state_panel()

    def _build_bp_gallery_panel_v2(self):
        header = ttk.Frame(self.bp_left)
        header.pack(fill=tk.X)
        ttk.Label(header, text="魔灵图库", font=("", 12, "bold")).pack(side=tk.LEFT)
        self.bp_gallery_count_label = ttk.Label(header, text="")
        self.bp_gallery_count_label.pack(side=tk.RIGHT)

        controls = ttk.LabelFrame(self.bp_left, text="BP 图库控制", padding=10, style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(8, 8))

        top_row = ttk.Frame(controls)
        top_row.pack(fill=tk.X)
        ttk.Label(top_row, text="先手").pack(side=tk.LEFT)
        ttk.Radiobutton(
            top_row,
            text="我方一选",
            value="me",
            variable=self.first_pick_var,
            command=self.refresh_all,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(
            top_row,
            text="敌方一选",
            value="enemy",
            variable=self.first_pick_var,
            command=self.refresh_all,
        ).pack(side=tk.LEFT, padx=(8, 0))

        filter_row = ttk.Frame(controls)
        filter_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(filter_row, text="搜索").pack(side=tk.LEFT)
        ttk.Entry(filter_row, textvariable=self.search_var, width=22).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(filter_row, text="属性").pack(side=tk.LEFT)
        self.bp_attr_filter_combo = ttk.Combobox(
            filter_row,
            textvariable=self.attr_filter_var,
            values=self.attributes,
            state="readonly",
            width=10,
        )
        self.bp_attr_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(filter_row, text="系列").pack(side=tk.LEFT)
        self.bp_family_filter_combo = ttk.Combobox(
            filter_row,
            textvariable=self.family_filter_var,
            values=self.families,
            state="readonly",
            width=16,
        )
        self.bp_family_filter_combo.pack(side=tk.LEFT, padx=(6, 0))

        check_row = ttk.Frame(controls)
        check_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            check_row,
            text="图库只看我的 Box",
            variable=self.only_owned_in_picker_var,
            command=self.refresh_bp_gallery,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            check_row,
            text="推荐只用我的 Box",
            variable=self.use_my_box_for_reco_var,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(
            check_row,
            text="只显示有图片",
            variable=self.only_with_image_var,
            command=self.refresh_bp_gallery,
        ).pack(side=tk.LEFT, padx=(12, 0))

        paging = ttk.Frame(self.bp_left)
        paging.pack(fill=tk.X, pady=(0, 8))
        self.bp_prev_btn = ttk.Button(paging, text="上一页", command=lambda: self.change_bp_page(-1))
        self.bp_prev_btn.pack(side=tk.LEFT)
        self.bp_page_label = ttk.Label(paging, text="第 1/1 页")
        self.bp_page_label.pack(side=tk.LEFT, padx=8)
        self.bp_next_btn = ttk.Button(paging, text="下一页", command=lambda: self.change_bp_page(1))
        self.bp_next_btn.pack(side=tk.LEFT)

        self.bp_gallery_scroll = ScrollableFrame(self.bp_left)
        self.bp_gallery_scroll.pack(fill=tk.BOTH, expand=True)

    def _build_bp_state_panel(self):
        self.bp_right_scroll = ScrollableFrame(self.bp_right)
        self.bp_right_scroll.pack(fill=tk.BOTH, expand=True)

    def _build_profile_tab(self):
        layout = ttk.Frame(self.profile_tab)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=4)
        layout.columnconfigure(1, weight=7)
        layout.rowconfigure(0, weight=1)

        left = ttk.Frame(layout, padding=(0, 0, 10, 0))
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(layout)
        right.grid(row=0, column=1, sticky="nsew")

        account_frame = ttk.LabelFrame(left, text="账号资料", padding=8, style="Section.TLabelframe")
        account_frame.pack(fill=tk.X, pady=(0, 8))

        def account_row(parent, label, variable, width=12):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=variable, width=width).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        account_row(account_frame, "账号名称", self.account_profile_name_var, width=18)
        account_row(account_frame, "抢速一速", self.account_contest_speed_1_var)
        account_row(account_frame, "抢速二速", self.account_contest_speed_2_var)
        account_row(account_frame, "高速剥条", self.account_fast_strip_speed_var)
        account_row(account_frame, "跟进控制速度", self.account_followup_cc_speed_var)
        account_row(account_frame, "跟进输出速度", self.account_followup_dps_speed_var)
        account_row(account_frame, "后手辅助速度", self.account_turn2_support_speed_var)

        confidence_row = ttk.Frame(account_frame)
        confidence_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(confidence_row, text="速度信心", width=20).pack(side=tk.LEFT)
        ttk.Combobox(confidence_row, textvariable=self.account_speed_confidence_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)
        ttk.Label(confidence_row, text="先手深度", width=12).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(confidence_row, textvariable=self.account_box_depth_speed_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)

        depth_row = ttk.Frame(account_frame)
        depth_row.pack(fill=tk.X, pady=2)
        ttk.Label(depth_row, text="后手肉队深度", width=20).pack(side=tk.LEFT)
        ttk.Combobox(depth_row, textvariable=self.account_box_depth_bruiser_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)
        ttk.Label(depth_row, text="反控深度", width=12).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(depth_row, textvariable=self.account_box_depth_anti_cc_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)

        tech_row = ttk.Frame(account_frame)
        tech_row.pack(fill=tk.X, pady=2)
        ttk.Label(tech_row, text="对策卡深度", width=20).pack(side=tk.LEFT)
        ttk.Combobox(tech_row, textvariable=self.account_box_depth_tech_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)

        account_row(account_frame, "备注", self.account_notes_var, width=18)

        build_frame = ttk.LabelFrame(left, text="魔灵构建", padding=8, style="Section.TLabelframe")
        build_frame.pack(fill=tk.BOTH, expand=True)
        self.profile_selected_label = ttk.Label(build_frame, text="请先在右侧图库选择魔灵")
        self.profile_selected_label.pack(anchor="w", pady=(0, 6))

        build_state_row = ttk.Frame(build_frame)
        build_state_row.pack(fill=tk.X, pady=2)
        ttk.Label(build_state_row, text="成型等级", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(build_state_row, from_=0, to=3, textvariable=self.monster_build_state_var, width=6).pack(side=tk.LEFT)
        ttk.Label(build_state_row, text="0-3", foreground="#666666").pack(side=tk.LEFT, padx=(6, 0))

        def build_row(parent, label, variable, width=16):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=variable, width=width).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        build_row(build_frame, "实际速度", self.monster_actual_speed_var)

        tier_row = ttk.Frame(build_frame)
        tier_row.pack(fill=tk.X, pady=2)
        ttk.Label(tier_row, text="符文档位", width=20).pack(side=tk.LEFT)
        ttk.Combobox(tier_row, textvariable=self.monster_rune_tier_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)
        ttk.Label(tier_row, text="神器档位", width=12).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(tier_row, textvariable=self.monster_artifact_tier_var, values=GRADE_OPTIONS, state="readonly", width=8).pack(side=tk.LEFT)

        ttk.Checkbutton(build_frame, text="技能已满", variable=self.monster_skillup_complete_var).pack(anchor="w", pady=(2, 2))
        build_row(build_frame, "上场信心(0-1)", self.monster_confidence_score_var)
        build_row(build_frame, "常用构建", self.monster_preferred_build_var)
        build_row(build_frame, "职能标签", self.monster_role_tags_var)
        build_row(build_frame, "备注", self.monster_notes_var, width=18)

        action_row = ttk.Frame(build_frame)
        action_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(action_row, text="粘贴导入账号", command=self.import_account_profile_from_clipboard).pack(side=tk.LEFT)
        ttk.Button(action_row, text="保存资料", command=self.save_profile_data).pack(side=tk.RIGHT)
        ttk.Button(action_row, text="删除当前", command=self.delete_selected_monster_build).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(action_row, text="重置表单", command=self.clear_selected_monster_build).pack(side=tk.RIGHT, padx=(0, 6))

        top = ttk.Frame(right)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="魔灵资料图库", font=("", 12, "bold")).pack(side=tk.LEFT)
        self.profile_count_label = ttk.Label(top, text="")
        self.profile_count_label.pack(side=tk.RIGHT)

        toolbar = ttk.Frame(right)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(toolbar, text="搜索").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.profile_search_var, width=20).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(toolbar, text="属性").pack(side=tk.LEFT)
        self.profile_attr_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.profile_attr_filter_var,
            values=self.attributes,
            state="readonly",
            width=10,
        )
        self.profile_attr_filter_combo.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(toolbar, text="系列").pack(side=tk.LEFT)
        self.profile_family_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.profile_family_filter_var,
            values=self.families,
            state="readonly",
            width=14,
        )
        self.profile_family_filter_combo.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Checkbutton(toolbar, text="仅看我的Box", variable=self.profile_only_box_var, command=self.refresh_profile_gallery).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="仅看已录入", variable=self.profile_only_saved_var, command=self.refresh_profile_gallery).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(toolbar, text="仅看有图片", variable=self.profile_only_with_image_var, command=self.refresh_profile_gallery).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(toolbar, text="重新载入", command=self.reload_profile_data).pack(side=tk.RIGHT)

        paging = ttk.Frame(right)
        paging.pack(fill=tk.X, pady=(0, 8))
        self.profile_prev_btn = ttk.Button(paging, text="上一页", command=lambda: self.change_profile_page(-1))
        self.profile_prev_btn.pack(side=tk.LEFT)
        self.profile_page_label = ttk.Label(paging, text="第 1/1 页")
        self.profile_page_label.pack(side=tk.LEFT, padx=8)
        self.profile_next_btn = ttk.Button(paging, text="下一页", command=lambda: self.change_profile_page(1))
        self.profile_next_btn.pack(side=tk.LEFT)

        self.profile_gallery_scroll = ScrollableFrame(right)
        self.profile_gallery_scroll.pack(fill=tk.BOTH, expand=True)

    def _apply_account_profile_to_vars(self, profile):
        profile = normalize_account_profile(profile)
        self.account_profile_name_var.set(profile.get("profile_name", ""))
        self.account_contest_speed_1_var.set("" if profile.get("contest_speed_1") is None else str(profile.get("contest_speed_1")))
        self.account_contest_speed_2_var.set("" if profile.get("contest_speed_2") is None else str(profile.get("contest_speed_2")))
        self.account_fast_strip_speed_var.set("" if profile.get("fast_strip_speed") is None else str(profile.get("fast_strip_speed")))
        self.account_followup_cc_speed_var.set("" if profile.get("followup_cc_speed") is None else str(profile.get("followup_cc_speed")))
        self.account_followup_dps_speed_var.set("" if profile.get("followup_dps_speed") is None else str(profile.get("followup_dps_speed")))
        self.account_turn2_support_speed_var.set("" if profile.get("turn2_support_speed") is None else str(profile.get("turn2_support_speed")))
        self.account_speed_confidence_var.set(profile.get("speed_confidence", "B"))
        self.account_box_depth_speed_var.set(profile.get("box_depth_speed", "B"))
        self.account_box_depth_bruiser_var.set(profile.get("box_depth_bruiser", "B"))
        self.account_box_depth_anti_cc_var.set(profile.get("box_depth_anti_cc", "B"))
        self.account_box_depth_tech_var.set(profile.get("box_depth_tech", "B"))
        self.account_notes_var.set(profile.get("notes", ""))

    def _load_profile_state_into_vars(self):
        self._apply_account_profile_to_vars(self.account_profile_state)
        self.clear_selected_monster_build()

    def _collect_account_profile_from_vars(self):
        return normalize_account_profile(
            {
                "profile_name": self.account_profile_name_var.get(),
                "contest_speed_1": self.account_contest_speed_1_var.get().strip(),
                "contest_speed_2": self.account_contest_speed_2_var.get().strip(),
                "fast_strip_speed": self.account_fast_strip_speed_var.get().strip(),
                "followup_cc_speed": self.account_followup_cc_speed_var.get().strip(),
                "followup_dps_speed": self.account_followup_dps_speed_var.get().strip(),
                "turn2_support_speed": self.account_turn2_support_speed_var.get().strip(),
                "speed_confidence": self.account_speed_confidence_var.get(),
                "box_depth_speed": self.account_box_depth_speed_var.get(),
                "box_depth_bruiser": self.account_box_depth_bruiser_var.get(),
                "box_depth_anti_cc": self.account_box_depth_anti_cc_var.get(),
                "box_depth_tech": self.account_box_depth_tech_var.get(),
                "notes": self.account_notes_var.get(),
            }
        )

    def _parse_account_profile_text(self, raw_text):
        text = str(raw_text or "").strip()
        if not text:
            return {}

        try:
            parsed_json = json.loads(text)
        except Exception:
            parsed_json = None

        if isinstance(parsed_json, dict):
            filtered = {}
            for raw_key, value in parsed_json.items():
                key = ACCOUNT_PROFILE_IMPORT_ALIASES.get(str(raw_key or "").strip().lower())
                if key:
                    filtered[key] = value
            if filtered:
                return normalize_account_profile(filtered)

        parsed = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            key_text = None
            value_text = None
            for separator in ("\t", "：", ":", "="):
                if separator in line:
                    key_text, value_text = line.split(separator, 1)
                    break

            if key_text is None:
                continue

            key = ACCOUNT_PROFILE_IMPORT_ALIASES.get(str(key_text or "").strip().lower())
            if not key:
                key = ACCOUNT_PROFILE_IMPORT_ALIASES.get(str(key_text or "").strip())
            if not key:
                continue
            parsed[key] = str(value_text or "").strip()

        return normalize_account_profile(parsed) if parsed else {}

    def import_account_profile_from_clipboard(self):
        try:
            raw_text = self.root.clipboard_get()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read clipboard text: {exc}")
            return False

        parsed = self._parse_account_profile_text(raw_text)
        if not parsed:
            messagebox.showinfo(
                "提示",
                "剪贴板里没有识别到账号资料。\n支持 JSON，或每行一个“字段: 值 / 字段\\t值”格式。",
            )
            return False

        merged = dict(self.account_profile_state)
        merged.update(parsed)
        self.account_profile_state = normalize_account_profile(merged)
        self._apply_account_profile_to_vars(self.account_profile_state)
        self.set_status("已从剪贴板导入账号资料，确认后可直接保存")
        return True

    def _current_monster_build_entry(self):
        if not self.profile_selected_monster:
            return None
        return normalize_monster_build_entry(
            self.profile_selected_monster,
            {
                "monster": self.profile_selected_monster,
                "build_state": self.monster_build_state_var.get(),
                "actual_speed": self.monster_actual_speed_var.get().strip(),
                "rune_tier": self.monster_rune_tier_var.get(),
                "artifact_tier": self.monster_artifact_tier_var.get(),
                "skillup_complete": bool(self.monster_skillup_complete_var.get()),
                "confidence_score": self.monster_confidence_score_var.get().strip(),
                "preferred_build": self.monster_preferred_build_var.get(),
                "role_tags": self.monster_role_tags_var.get(),
                "notes": self.monster_notes_var.get(),
            },
        )

    def _load_monster_build_into_form(self, monster):
        self.profile_selected_monster = monster
        entry = self.monster_builds_state.get(monster) or make_default_monster_build(monster)
        entry = normalize_monster_build_entry(monster, entry)
        self.monster_build_state_var.set(int(entry.get("build_state", 0)))
        self.monster_actual_speed_var.set("" if entry.get("actual_speed") is None else str(entry.get("actual_speed")))
        self.monster_rune_tier_var.set(entry.get("rune_tier", "C"))
        self.monster_artifact_tier_var.set(entry.get("artifact_tier", "C"))
        self.monster_skillup_complete_var.set(bool(entry.get("skillup_complete", False)))
        self.monster_confidence_score_var.set(str(entry.get("confidence_score", 0.0)))
        self.monster_preferred_build_var.set(entry.get("preferred_build", ""))
        self.monster_role_tags_var.set(", ".join(entry.get("role_tags", [])))
        self.monster_notes_var.set(entry.get("notes", ""))
        in_box = monster in set(self.my_box_state)
        state_label = self._profile_build_state_label(int(entry.get("build_state", 0)))
        self.profile_selected_label.configure(
            text=f"{monster} | 在Box: {'是' if in_box else '否'} | 成型: {state_label}"
        )

    def _profile_build_state_label(self, build_state):
        mapping = {
            0: "未录入",
            1: "半成型",
            2: "可上场",
            3: "核心成型",
        }
        return mapping.get(int(build_state), "未录入")

    def select_profile_monster(self, monster):
        self._load_monster_build_into_form(monster)

    def clear_selected_monster_build(self):
        if self.profile_selected_monster:
            monster = self.profile_selected_monster
            self._load_monster_build_into_form(monster)
            default_entry = make_default_monster_build(monster)
            self.monster_build_state_var.set(int(default_entry["build_state"]))
            self.monster_actual_speed_var.set("")
            self.monster_rune_tier_var.set(default_entry["rune_tier"])
            self.monster_artifact_tier_var.set(default_entry["artifact_tier"])
            self.monster_skillup_complete_var.set(bool(default_entry["skillup_complete"]))
            self.monster_confidence_score_var.set(str(default_entry["confidence_score"]))
            self.monster_preferred_build_var.set("")
            self.monster_role_tags_var.set("")
            self.monster_notes_var.set("")
            self.profile_selected_label.configure(text=f"{monster} | 表单已重置")
            return

        self.monster_build_state_var.set(0)
        self.monster_actual_speed_var.set("")
        self.monster_rune_tier_var.set("C")
        self.monster_artifact_tier_var.set("C")
        self.monster_skillup_complete_var.set(False)
        self.monster_confidence_score_var.set("0.0")
        self.monster_preferred_build_var.set("")
        self.monster_role_tags_var.set("")
        self.monster_notes_var.set("")
        if hasattr(self, "profile_selected_label"):
            self.profile_selected_label.configure(text="请先在右侧图库选择魔灵")

    def delete_selected_monster_build(self):
        if not self.profile_selected_monster:
            messagebox.showinfo("提示", "请先选择一个魔灵。")
            return
        if self.profile_selected_monster in self.monster_builds_state:
            self.monster_builds_state.pop(self.profile_selected_monster, None)
        self.clear_selected_monster_build()
        self.refresh_profile_gallery()
        self.set_status(f"已删除 {self.profile_selected_monster} 的构建资料")

    def save_profile_data(self):
        self.account_profile_state = self._collect_account_profile_from_vars()
        if self.profile_selected_monster:
            entry = self._current_monster_build_entry()
            if entry is not None:
                is_empty = (
                    int(entry.get("build_state", 0)) <= 0
                    and entry.get("actual_speed") is None
                    and not entry.get("preferred_build")
                    and not entry.get("role_tags")
                    and not entry.get("notes")
                    and not entry.get("skillup_complete")
                    and float(entry.get("confidence_score", 0.0)) <= 0.0
                )
                if is_empty:
                    self.monster_builds_state.pop(self.profile_selected_monster, None)
                else:
                    self.monster_builds_state[self.profile_selected_monster] = entry
        self.monster_builds_state = normalize_monster_builds(self.monster_builds_state)
        save_json_file(ACCOUNT_PROFILE_PATH, self.account_profile_state)
        save_json_file(MONSTER_BUILDS_PATH, self.monster_builds_state)
        self.refresh_profile_gallery()
        self.set_status(
            f"已保存账号资料与 {len(self.monster_builds_state)} 条魔灵构建记录"
        )

    def reload_profile_data(self):
        self.account_profile_state = normalize_account_profile(load_json_file(ACCOUNT_PROFILE_PATH, ACCOUNT_PROFILE_DEFAULT))
        self.monster_builds_state = normalize_monster_builds(load_json_file(MONSTER_BUILDS_PATH, {}))
        self._load_profile_state_into_vars()
        self.refresh_profile_gallery()
        self.set_status(
            f"已重新载入账号资料与 {len(self.monster_builds_state)} 条魔灵构建记录"
        )

    def change_profile_page(self, delta):
        self.profile_gallery_page = max(1, self.profile_gallery_page + delta)
        self.refresh_profile_gallery()

    def refresh_profile_tab(self):
        if not self._core_data_ready:
            self.refresh_profile_gallery()
            return
        self._profile_tab_loaded = True
        self.refresh_profile_gallery()

    def refresh_profile_gallery(self):
        if not hasattr(self, "profile_gallery_scroll"):
            return
        if not self._core_data_ready:
            self.profile_gallery_scroll.clear()
            ttk.Label(self.profile_gallery_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            return

        monsters = self._filter_monsters(
            search_text=self.profile_search_var.get(),
            attr_filter=self.profile_attr_filter_var.get(),
            family_filter=self.profile_family_filter_var.get(),
            only_owned=self.profile_only_box_var.get(),
            only_with_image=self.profile_only_with_image_var.get(),
        )
        if self.profile_only_saved_var.get():
            saved_names = set(self.monster_builds_state)
            monsters = [monster for monster in monsters if monster in saved_names]

        raw_page_size = max(BP_GALLERY_COLUMNS, int(self.profile_page_size_var.get()))
        page_size = int(math.ceil(raw_page_size / BP_GALLERY_COLUMNS) * BP_GALLERY_COLUMNS)
        page_items, current_page, total_pages, total_items = self.paginate(monsters, self.profile_gallery_page, page_size)
        self.profile_gallery_page = current_page
        if hasattr(self, "profile_count_label"):
            self.profile_count_label.configure(
                text=f"已录入: {len(self.monster_builds_state)} | 图库总数: {total_items}"
            )
        if hasattr(self, "profile_page_label"):
            self.profile_page_label.configure(text=f"第 {current_page}/{total_pages} 页")
        if hasattr(self, "profile_prev_btn"):
            self.profile_prev_btn.configure(state=tk.NORMAL if current_page > 1 else tk.DISABLED)
        if hasattr(self, "profile_next_btn"):
            self.profile_next_btn.configure(state=tk.NORMAL if current_page < total_pages else tk.DISABLED)

        self.profile_gallery_scroll.clear()
        columns = BP_GALLERY_COLUMNS
        owned_set = set(self.my_box_state)
        for idx, monster in enumerate(page_items):
            row = idx // columns
            col = idx % columns
            entry = self.monster_builds_state.get(monster)
            build_state = 0 if not entry else int(entry.get("build_state", 0))
            badge_text = self._profile_build_state_label(build_state) if entry else ""
            subtitle_parts = []
            if entry and entry.get("actual_speed") is not None:
                subtitle_parts.append(f"速度 {entry['actual_speed']}")
            if entry:
                subtitle_parts.append(f"符文 {entry.get('rune_tier', 'C')}")
                subtitle_parts.append(f"神器 {entry.get('artifact_tier', 'C')}")
            subtitle = "\n".join(subtitle_parts[:2])
            card = self.render_image_card(
                self.profile_gallery_scroll.inner,
                monster,
                size=BP_GALLERY_IMAGE_SIZE,
                subtitle=subtitle,
                badge_text=badge_text,
                owned=monster in owned_set,
                card_command=lambda m=monster: self.select_profile_monster(m),
            )
            card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            self.profile_gallery_scroll.inner.grid_columnconfigure(col, weight=1)

    def _build_profile_tab(self):
        for child in self.profile_tab.winfo_children():
            child.destroy()
        self._build_comp_recommendation_tab()

    def _build_comp_recommendation_tab(self):
        layout = ttk.Frame(self.profile_tab)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=4)
        layout.columnconfigure(1, weight=7)
        layout.rowconfigure(0, weight=1)

        control = ttk.Frame(layout, padding=(0, 0, 10, 0))
        control.grid(row=0, column=0, sticky="nsew")
        result = ttk.Frame(layout)
        result.grid(row=0, column=1, sticky="nsew")

        controls = ttk.LabelFrame(control, text="阵容推荐控制", padding=10, style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(0, 8))

        season_row = ttk.Frame(controls)
        season_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(season_row, text="样本范围", width=10).pack(side=tk.LEFT)
        self.comp_season_combo = ttk.Combobox(
            season_row,
            textvariable=self.comp_reco_season_var,
            values=["全部样本"],
            state="readonly",
            width=20,
        )
        self.comp_season_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mode_row = ttk.Frame(controls)
        mode_row.pack(fill=tk.X)
        ttk.Label(mode_row, text="推荐类型", width=10).pack(side=tk.LEFT)
        self.comp_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.comp_reco_mode_var,
            values=[
                "当前赛季强势三系阵容",
                "基于核心补全阵容",
                "当前赛季最强阵容",
            ],
            state="readonly",
            width=24,
        )
        self.comp_mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        core_frame = ttk.LabelFrame(control, text="核心魔灵选择", padding=10, style="Section.TLabelframe")
        core_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        search_row = ttk.Frame(core_frame)
        search_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(search_row, text="搜索", width=8).pack(side=tk.LEFT)
        ttk.Entry(search_row, textvariable=self.comp_core_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(search_row, text="加入核心", command=self.add_selected_core_monster).pack(side=tk.RIGHT)

        self.comp_core_listbox = tk.Listbox(core_frame, height=10, exportselection=False)
        self.comp_core_listbox.pack(fill=tk.BOTH, expand=True)
        self.comp_core_listbox.bind("<Double-Button-1>", lambda _event: self.add_selected_core_monster())

        selected_header = ttk.Frame(core_frame)
        selected_header.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(selected_header, text="已选核心", font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(selected_header, text="清空核心", command=self.clear_core_monsters).pack(side=tk.RIGHT)

        self.comp_core_selected_frame = ttk.Frame(core_frame)
        self.comp_core_selected_frame.pack(fill=tk.X)

        options = ttk.LabelFrame(control, text="生成选项", padding=10, style="Section.TLabelframe")
        options.pack(fill=tk.X)

        topk_row = ttk.Frame(options)
        topk_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(topk_row, text="推荐数量", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(topk_row, from_=3, to=20, textvariable=self.comp_reco_top_k_var, width=8).pack(side=tk.LEFT)

        ttk.Checkbutton(
            options,
            text="只考虑我的 Box",
            variable=self.comp_reco_use_box_var,
            command=self._invalidate_comp_recommendations,
        ).pack(anchor="w")

        self.comp_generate_btn = ttk.Button(control, text="生成推荐", command=self.generate_comp_recommendations)
        self.comp_generate_btn.pack(fill=tk.X, pady=(10, 0))

        result_top = ttk.Frame(result)
        result_top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(result_top, text="阵容推荐结果", font=("", 12, "bold")).pack(side=tk.LEFT)
        self.comp_result_count_label = ttk.Label(result_top, text="", foreground="#666666")
        self.comp_result_count_label.pack(side=tk.RIGHT)

        self.comp_results_scroll = ScrollableFrame(result)
        self.comp_results_scroll.pack(fill=tk.BOTH, expand=True)

        self.refresh_profile_tab()

    def _invalidate_comp_recommendations(self, *_args):
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def _get_comp_recommender(self):
        raise RuntimeError("Legacy BPCompRecommender has been removed.")

    def _refresh_comp_season_options(self):
        if not hasattr(self, "comp_season_combo"):
            return
        options = ["全部样本"]
        if self._core_data_ready:
            options = self._get_comp_recommender().list_seasons()
        self.comp_season_combo.configure(values=options)
        if self.comp_reco_season_var.get() not in options:
            self.comp_reco_season_var.set(options[0])

    def _current_comp_season(self):
        value = self.comp_reco_season_var.get().strip()
        return None if not value or value == "全部样本" else value

    def refresh_comp_core_candidates(self):
        if not hasattr(self, "comp_core_listbox"):
            return
        self.comp_core_listbox.delete(0, tk.END)
        if not self._core_data_ready:
            self.comp_core_listbox.insert(tk.END, "Core data is loading...")
            return

        search_text = self.comp_core_search_var.get().strip().lower()
        selected = {monster for monster in self.comp_core_monsters}
        monsters = list(self.monster_pool)
        if self.comp_reco_use_box_var.get():
            owned = set(self.my_box_state)
            monsters = [monster for monster in monsters if monster in owned]
        if search_text:
            monsters = [monster for monster in monsters if search_text in monster.lower()]
        monsters = [monster for monster in sort_monsters_by_pick_frequency(monsters, self.pick_freq) if monster not in selected]
        for monster in monsters[:250]:
            self.comp_core_listbox.insert(tk.END, monster)

    def refresh_selected_core_monsters(self):
        if not hasattr(self, "comp_core_selected_frame"):
            return
        self.clear_children(self.comp_core_selected_frame)
        if not self.comp_core_monsters:
            ttk.Label(self.comp_core_selected_frame, text="未选择核心时，也可以直接推荐通用强势阵容。", foreground="#666666").pack(anchor="w")
            return
        columns = 3
        for idx, monster in enumerate(self.comp_core_monsters):
            card = self.render_image_card(
                self.comp_core_selected_frame,
                monster,
                size=(64, 64),
                subtitle=self.attr_map.get(monster, ""),
                primary_buttons=[("移除", lambda m=monster: self.remove_core_monster(m))],
                owned=monster in set(self.my_box_state),
            )
            row = idx // columns
            col = idx % columns
            card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            self.comp_core_selected_frame.grid_columnconfigure(col, weight=1)

    def add_selected_core_monster(self):
        if not hasattr(self, "comp_core_listbox"):
            return
        selection = self.comp_core_listbox.curselection()
        if not selection:
            return
        monster = self.comp_core_listbox.get(selection[0])
        self.add_core_monster(monster)

    def add_core_monster(self, monster):
        monster = str(monster or "").strip()
        if not monster or monster in self.comp_core_monsters:
            return
        if len(self.comp_core_monsters) >= 5:
            self.set_status("Core is already full")
            return
        self.comp_core_monsters.append(monster)
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def remove_core_monster(self, monster):
        self.comp_core_monsters = [item for item in self.comp_core_monsters if item != monster]
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def clear_core_monsters(self):
        self.comp_core_monsters = []
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def set_core_monsters_from_comp(self, monsters):
        self.comp_core_monsters = list(monsters[:5])
        self.comp_reco_mode_var.set("基于核心补全阵容")
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def refresh_profile_tab(self):
        self._profile_tab_loaded = True
        self._refresh_comp_season_options()
        self.refresh_comp_core_candidates()
        self.refresh_selected_core_monsters()
        self.refresh_comp_recommendation_results()
        if hasattr(self, "comp_generate_btn"):
            self.comp_generate_btn.configure(state=tk.NORMAL if self._core_data_ready else tk.DISABLED)

    def generate_comp_recommendations(self):
        if not self._ensure_core_data_ready():
            return

        mode = self.comp_reco_mode_var.get().strip()
        top_k = max(1, int(self.comp_reco_top_k_var.get() or 10))
        use_my_box = bool(self.comp_reco_use_box_var.get())
        season = self._current_comp_season()
        core = list(self.comp_core_monsters)

        def work():
            recommender = self._get_comp_recommender()
            if mode == "当前赛季强势三系阵容":
                return recommender.recommend_meta_comps(season=season, top_k=top_k, use_my_box=use_my_box)
            if mode == "基于核心补全阵容":
                return recommender.complete_comp_from_core(core, season=season, top_k=top_k, use_my_box=use_my_box)
            return recommender.recommend_best_comps(season=season, top_k=top_k, use_my_box=use_my_box)

        def apply_result(plans):
            self.comp_recommendations = list(plans or [])
            self.refresh_comp_recommendation_results()
            self.set_status(f"Generated {len(self.comp_recommendations)} comp recommendations")

        return self._start_background_task("Generating comp recommendations", work, apply_result)

    def copy_comp_to_clipboard(self, comp):
        text = ", ".join(comp or [])
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.set_status(f"Copied comp: {text}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to copy comp: {exc}")

    def refresh_comp_recommendation_results(self):
        if not hasattr(self, "comp_results_scroll"):
            return
        self.comp_results_scroll.clear()
        if not self._core_data_ready:
            ttk.Label(self.comp_results_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            if hasattr(self, "comp_result_count_label"):
                self.comp_result_count_label.configure(text="")
            return
        if not self.comp_recommendations:
            hint = "先选择推荐模式，再点击“生成推荐”。"
            if self.comp_reco_mode_var.get() == "基于核心补全阵容" and self.comp_core_monsters:
                hint = "核心已更新，点击“生成推荐”刷新补全结果。"
            ttk.Label(self.comp_results_scroll.inner, text=hint, foreground="#666666").pack(anchor="w")
            if hasattr(self, "comp_result_count_label"):
                self.comp_result_count_label.configure(text="0 results")
            return
        if hasattr(self, "comp_result_count_label"):
            self.comp_result_count_label.configure(text=f"{len(self.comp_recommendations)} results")
        for item in self.comp_recommendations:
            self._render_comp_recommendation_card(item)

    def _render_comp_recommendation_card(self, item):
        card = ttk.Frame(self.comp_results_scroll.inner, padding=10, style="Card.TFrame")
        card.pack(fill=tk.X, pady=(0, 10))

        top = ttk.Frame(card)
        top.pack(fill=tk.X)
        ttk.Label(top, text=item.get("name", "阵容"), font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            top,
            text=f"Score {float(item.get('score', 0.0)):.4f}",
            foreground="#3153a1",
            font=("", 10, "bold"),
        ).pack(side=tk.RIGHT)

        style_text = item.get("style", "balanced")
        detail = item.get("detail", {}) or {}
        summary = f"{style_text} | support {int(detail.get('support', 0) or 0)}"
        if detail.get("win_rate") is not None:
            summary += f" | win {float(detail.get('win_rate', 0.0)):.1%}"
        ttk.Label(card, text=summary, foreground="#555555").pack(anchor="w", pady=(2, 6))

        image_row = ttk.Frame(card)
        image_row.pack(fill=tk.X, pady=(0, 8))
        for idx, monster in enumerate(item.get("comp", [])):
            monster_card = self.render_image_card(
                image_row,
                monster,
                size=(68, 68),
                subtitle=self.attr_map.get(monster, ""),
                owned=monster in set(self.my_box_state),
            )
            monster_card.grid(row=0, column=idx, padx=3, pady=2, sticky="nsew")
            image_row.grid_columnconfigure(idx, weight=1)

        for reason in item.get("reasons", []):
            ttk.Label(card, text=f"- {reason}", wraplength=820, justify=tk.LEFT).pack(anchor="w")

        action_row = ttk.Frame(card)
        action_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(action_row, text="复制阵容", command=lambda comp=item.get("comp", []): self.copy_comp_to_clipboard(comp)).pack(side=tk.LEFT)
        ttk.Button(action_row, text="设为当前核心", command=lambda comp=item.get("comp", []): self.set_core_monsters_from_comp(comp)).pack(side=tk.LEFT, padx=(8, 0))

    def _build_box_tab(self):
        top = ttk.Frame(self.box_tab)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="我的 Box 管理", font=("", 12, "bold")).pack(side=tk.LEFT)
        self.box_count_label = ttk.Label(top, text="")
        self.box_count_label.pack(side=tk.RIGHT)

        toolbar = ttk.Frame(self.box_tab)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(toolbar, text="搜索").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.box_search_var, width=24).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(toolbar, text="属性").pack(side=tk.LEFT)
        self.box_attr_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.box_attr_filter_var,
            values=self.attributes,
            state="readonly",
            width=10,
        )
        self.box_attr_filter_combo.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Checkbutton(toolbar, text="只显示已拥有", variable=self.box_only_owned_var, command=self.refresh_box_tab).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="只显示有图片", variable=self.box_only_with_image_var, command=self.refresh_box_tab).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(toolbar, text="保存 Box", command=self.save_my_box).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="清空 Box", command=self.clear_my_box).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="重新载入", command=self.reload_my_box).pack(side=tk.RIGHT, padx=(0, 6))

        paging = ttk.Frame(self.box_tab)
        paging.pack(fill=tk.X, pady=(0, 8))
        self.box_prev_btn = ttk.Button(paging, text="上一页", command=lambda: self.change_box_page(-1))
        self.box_prev_btn.pack(side=tk.LEFT)
        self.box_page_label = ttk.Label(paging, text="第 1/1 页")
        self.box_page_label.pack(side=tk.LEFT, padx=8)
        self.box_next_btn = ttk.Button(paging, text="下一页", command=lambda: self.change_box_page(1))
        self.box_next_btn.pack(side=tk.LEFT)

        body = ttk.Frame(self.box_tab)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=7)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        current_box_frame = ttk.LabelFrame(left, text="当前 Box 已有魔灵", padding=8, style="Section.TLabelframe")
        current_box_frame.pack(fill=tk.BOTH, pady=(0, 8))
        self.current_box_scroll = ScrollableFrame(current_box_frame)
        self.current_box_scroll.pack(fill=tk.BOTH, expand=True)
        self.current_box_scroll.canvas.configure(height=310)

        self.box_gallery_scroll = ScrollableFrame(left)
        self.box_gallery_scroll.pack(fill=tk.BOTH, expand=True)

        self.box_reco_section = ttk.LabelFrame(right, text="补强推荐", padding=8, style="Section.TLabelframe")
        self.box_reco_section.pack(fill=tk.BOTH, expand=True)
        self.box_reco_scroll = ScrollableFrame(self.box_reco_section)
        self.box_reco_scroll.pack(fill=tk.BOTH, expand=True)
        self.box_reco_scroll.canvas.configure(height=620)

    def _bind_var_events(self):
        self.first_pick_var.trace_add("write", self.on_first_pick_side_changed)

        for variable in [
            self.search_var,
            self.attr_filter_var,
            self.family_filter_var,
            self.bp_page_size_var,
        ]:
            variable.trace_add("write", lambda *_: self.on_bp_filter_changed())

        for variable in [
            self.box_search_var,
            self.box_attr_filter_var,
            self.box_page_size_var,
        ]:
            variable.trace_add("write", lambda *_: self.on_box_filter_changed())

        self.comp_core_search_var.trace_add("write", lambda *_: self.refresh_profile_tab())
        for variable in [
            self.comp_reco_mode_var,
            self.comp_reco_season_var,
            self.comp_reco_top_k_var,
        ]:
            variable.trace_add("write", lambda *_: self.on_profile_filter_changed())

        self.bp_enemy_candidate_top_n_var.trace_add("write", lambda *_: self.refresh_bp_state_panel())

    def on_bp_filter_changed(self):
        self.bp_gallery_page = 1
        self.refresh_bp_gallery()

    def on_box_filter_changed(self):
        self.box_gallery_page = 1
        if self._box_tab_loaded or self._current_tab_name() == "box":
            self.refresh_box_tab()
        if self._profile_tab_loaded:
            self.refresh_profile_tab()

    def on_profile_filter_changed(self):
        if self._profile_tab_loaded or self._current_tab_name() == "profile":
            self._invalidate_comp_recommendations()

    def get_photo(self, monster, size=(96, 96)):
        key = (monster, size)
        if key in self.photo_cache:
            return self.photo_cache[key]

        image_path = get_image_path_for_monster(monster, self.image_lookup)
        if not image_path:
            self.photo_cache[key] = None
            return None

        try:
            image = Image.open(image_path).convert("RGBA")
            image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
        except Exception:
            photo = None

        self.photo_cache[key] = photo
        return photo

    def set_status(self, text):
        self.status_var.set(text)
        self.last_action_var.set(text)

    def _set_transient_status(self, text):
        self.status_var.set(text)

    def _relation_model_bundle_ready(self):
        return (
            (ROOT_DIR / "models" / "bp_relation_net.pt").exists()
            and (ROOT_DIR / "models" / "bp_relation_meta.json").exists()
            and (ROOT_DIR / "models" / "bp_relation_vocab.json").exists()
        )

    def _get_relation_recommender(self):
        if self.relation_recommender is not None:
            return self.relation_recommender
        if not self._relation_model_bundle_ready():
            raise RuntimeError("Relation model is not available. Run python train_bp_relation_net.py first.")
        from bp_relation_recommender import BPRelationRecommender

        self.relation_recommender = BPRelationRecommender()
        return self.relation_recommender

    def _get_ban_value_recommender(self):
        if self.ban_value_recommender is not None:
            return self.ban_value_recommender
        from bp_ban_value_recommender import BPBanValueRecommender

        self.ban_value_recommender = BPBanValueRecommender()
        return self.ban_value_recommender

    def _get_bp_analyzer(self):
        if self.bp_analyzer is not None:
            return self.bp_analyzer
        from sw_ui_detection.bp_screen_analyzer import BPDraftScreenAnalyzer

        self.bp_analyzer = BPDraftScreenAnalyzer()
        return self.bp_analyzer

    def _recommendation_context_signature(self):
        return (
            tuple(self.my_picks),
            tuple(self.enemy_picks),
            tuple(self.banned_monsters),
            self.my_final_ban or "",
            self.enemy_final_ban or "",
            self.first_pick_var.get(),
            self.my_leader_var.get().strip(),
            self.enemy_leader_var.get().strip(),
            bool(self.use_my_box_for_reco_var.get()),
            tuple(self.my_box_state),
            int(self.top_k_var.get()),
        )

    def _clear_recommendation_results(self):
        self.pick_plans = []
        self.enemy_predictions = []
        self.ban_plans = []

    def _current_pick_order(self):
        order = len(self.my_picks) + len(self.enemy_picks) + 1
        if order < 1 or order > 10:
            return None
        return order

    def _pick_side_for_order(self, order):
        if order is None or order < 1 or order > 10:
            return None
        first = "my" if normalize_name(self.first_pick_var.get()) != "enemy" else "enemy"
        other = "enemy" if first == "my" else "my"
        pattern = [first, other, other, first, first, other, other, first, first, other]
        return pattern[order - 1]

    def _current_pick_side(self):
        return self._pick_side_for_order(self._current_pick_order())

    def _current_pick_span(self):
        order = self._current_pick_order()
        side = self._pick_side_for_order(order)
        if order is None or side is None:
            return None, None

        count = 1
        while self._pick_side_for_order(order + count) == side:
            count += 1
        return order, count

    def _effective_bp_page_size(self):
        raw_size = max(BP_GALLERY_COLUMNS, int(self.bp_page_size_var.get()))
        return int(math.ceil(raw_size / BP_GALLERY_COLUMNS) * BP_GALLERY_COLUMNS)

    def _get_post_ban_final_picks(self):
        enemy_banned_key = normalize_name(self.my_final_ban)
        my_banned_key = normalize_name(self.enemy_final_ban)
        enemy_final_picks = [m for m in self.enemy_picks if normalize_name(m) != enemy_banned_key]
        my_final_picks = [m for m in self.my_picks if normalize_name(m) != my_banned_key]
        return my_final_picks, enemy_final_picks

    def _predict_enemy_ban_target(self):
        if len(self.my_picks) != 5 or len(self.enemy_picks) != 5:
            return None

        best_target = None
        best_wr = None
        ban_value_recommender = self._get_ban_value_recommender()
        for monster in self.my_picks:
            result = ban_value_recommender.evaluate_state(
                my_picks=self.my_picks,
                enemy_picks=self.enemy_picks,
                my_ban=self.my_final_ban,
                enemy_ban=monster,
                my_leader=self.my_leader_var.get().strip(),
                enemy_leader=self.enemy_leader_var.get().strip(),
                return_detail=True,
            )
            win_rate = float(result.get("calibrated_win_rate", 0.0))
            if best_wr is None or win_rate < best_wr:
                best_wr = win_rate
                best_target = monster

        if best_target is None:
            return None
        return {
            "monster": best_target,
            "estimated_win_rate": best_wr,
        }

    def _get_box_focus_monsters(self, limit=10):
        current_box = sort_monsters_by_pick_frequency(self.my_box_state, self.pick_freq)
        return current_box[: max(1, int(limit))]

    def _analyze_box_collection(self):
        focus_box = self._get_box_focus_monsters(limit=10)
        if not focus_box:
            return {
                "focus_box": [],
                "core_style": "未知",
                "missing_tags": [],
                "top_tags": [],
                "attr_summary": [],
                "family_summary": [],
            }

        if self.recommender is None:
            return {
                "focus_box": focus_box,
                "core_style": "loading",
                "missing_tags": [],
                "top_tags": [],
                "attr_summary": [],
                "family_summary": [],
            }

        profile = self.recommender.debug_comp_profile(focus_box)
        tag_count = profile.get("tag_count", {})
        top_tags = [tag for tag, count in sorted(tag_count.items(), key=lambda item: (-item[1], item[0])) if count > 0][:6]
        attr_summary = [name for name, _count in Counter(self.attr_map.get(m, "") for m in self.my_box_state if self.attr_map.get(m)).most_common(3)]
        family_summary = [name for name, _count in Counter(self.family_map.get(m, "") for m in self.my_box_state if self.family_map.get(m)).most_common(3)]
        return {
            "focus_box": focus_box,
            "core_style": profile.get("core_style") or "未知",
            "missing_tags": profile.get("missing_tags", []),
            "top_tags": top_tags,
            "attr_summary": attr_summary,
            "family_summary": family_summary,
        }

    def _build_box_reinforcement_candidates(self, limit=6):
        if self.recommender is None:
            return []
        owned = {normalize_name(monster) for monster in self.my_box_state if normalize_name(monster)}
        profile = self.recommender._analyze_comp_profile(self._get_box_focus_monsters(limit=10))
        suggestions = []

        for monster in self.monster_pool:
            if normalize_name(monster) in owned:
                continue

            need_score, need_reasons = self.recommender._my_need_score(profile, monster)
            syn_score, syn_reasons = self.recommender._synergy_score(profile, monster)
            style_tags = sorted(list(self.recommender.get_monster_style_tags(monster)))
            cover_tags = [tag for tag in style_tags if tag in profile.get("missing_tags", [])]
            popularity_bonus = min(1.2, math.log1p(max(0, self.pick_freq.get(monster, 0))) * 0.18)
            score = need_score * 1.45 + syn_score * 0.85 + popularity_bonus

            reasons = []
            reasons.extend(need_reasons[:2])
            reasons.extend(syn_reasons[:1])
            if not reasons and cover_tags:
                reasons.append("补足：" + " / ".join(cover_tags[:2]))
            if not reasons and style_tags:
                reasons.append("风格：" + " / ".join(style_tags[:2]))
            if not reasons and self.pick_freq.get(monster, 0) > 0:
                reasons.append("高频泛用补强")

            suggestions.append(
                {
                    "monster": monster,
                    "score": round(score, 4),
                    "reasons": reasons[:2],
                    "style_tags": style_tags[:3],
                }
            )

        suggestions.sort(key=lambda item: (item["score"], self.pick_freq.get(item["monster"], 0), item["monster"]), reverse=True)
        return suggestions[: max(1, int(limit))]

    def _box_gallery_primary_buttons(self, monster, owned):
        if owned:
            return [("Remove", lambda m=monster: self.toggle_my_box(m))]
        return [("Add", lambda m=monster: self.toggle_my_box(m))]

    def refresh_box_recommendations(self):
        if not hasattr(self, "box_reco_scroll"):
            return

        self.box_reco_scroll.clear()
        ttk.Label(
            self.box_reco_scroll.inner,
            text="Box recommendation was removed. This panel no longer loads historical samples.",
            foreground="#666666",
            wraplength=260,
            justify=tk.LEFT,
        ).pack(anchor="w")
        return
        if not self._core_data_ready:
            ttk.Label(self.box_reco_scroll.inner, text="Recommendations will appear after core data loads").pack(anchor="w")
            return

        if not self.my_box_state:
            ttk.Label(
                self.box_reco_scroll.inner,
                text="Box 为空，先从左侧图库加入一些常用魔灵。",
                foreground="#666666",
                wraplength=260,
                justify=tk.LEFT,
            ).pack(anchor="w")
            return

        suggestions = self._build_box_reinforcement_candidates(limit=6)
        if not suggestions:
            ttk.Label(self.box_reco_scroll.inner, text="暂无补强推荐").pack(anchor="w")
            return

        for idx, item in enumerate(suggestions, start=1):
            monster = item["monster"]
            reason_text = "；".join(item.get("reasons", [])[:2]) or "点击加入 Box"
            subtitle = f"得分 {float(item.get('score', 0.0)):.2f}\n{reason_text}"
            card = self.render_image_card(
                self.box_reco_scroll.inner,
                monster,
                size=(78, 78),
                subtitle=subtitle,
                badge_text=f"补强 {idx}",
                primary_buttons=[("Add", lambda m=monster: self.toggle_my_box(m))],
            )
            card.pack(fill=tk.X, pady=4)

    def _auto_recommendation_type(self):
        if len(self.my_picks) == 5 and len(self.enemy_picks) == 5:
            return "ban" if not self.my_final_ban else None

        stage = infer_bp_stage(self.my_picks, self.enemy_picks)
        if stage in {"pick_single", "pick_multi"}:
            side = self._current_pick_side()
            if side == "my":
                return "pick"
            if side == "enemy":
                return "enemy_prediction"
        return None

    def _run_auto_recommendation_if_needed(self):
        if not self._core_data_ready:
            return False
        action = self._auto_recommendation_type()
        self._pending_auto_recommend = False
        if not action:
            return False

        if self._active_task_name:
            self._pending_auto_recommend = True
            return False

        if action == "pick":
            return self.generate_pick_recommendations(silent=True)
        if action == "enemy_prediction":
            return self.generate_enemy_predictions(silent=True)
        return self.generate_ban_recommendations(silent=True)

    def _apply_bp_state_change(self, status_text):
        self._clear_recommendation_results()
        self.set_status(status_text)
        self.refresh_all()
        self._run_auto_recommendation_if_needed()

    def on_first_pick_side_changed(self, *_args):
        self._clear_recommendation_results()
        self.set_status("已切换先手设置")
        self.root.after(0, self._run_auto_recommendation_if_needed)

    def _clear_active_task(self, task_id):
        if task_id != self._active_task_id:
            return
        self._active_task_name = None
        if self._active_task_status_job is not None:
            try:
                self.root.after_cancel(self._active_task_status_job)
            except Exception:
                pass
            self._active_task_status_job = None

    def _pulse_active_task_status(self, task_id, task_name, step=0):
        if task_id != self._active_task_id or self._active_task_name != task_name:
            return

        dots = "." * ((step % 3) + 1)
        self._set_transient_status(f"{task_name}{dots}")
        self._active_task_status_job = self.root.after(
            400,
            lambda: self._pulse_active_task_status(task_id, task_name, step + 1),
        )

    def _poll_background_task(self, task_id, task_name, apply_result_fn):
        if task_id != self._active_task_id or self._active_task_name != task_name:
            return

        try:
            finished_task_id, result, error, error_trace = self._task_result_queue.get_nowait()
        except queue.Empty:
            self.root.after(
                100,
                lambda: self._poll_background_task(task_id, task_name, apply_result_fn),
            )
            return

        if finished_task_id != task_id:
            self.root.after(
                0,
                lambda: self._poll_background_task(task_id, task_name, apply_result_fn),
            )
            return

        self._clear_active_task(task_id)
        if error is not None:
            print(error_trace, file=sys.stderr)
            self.set_status(f"{task_name} failed: {error}")
            messagebox.showerror("Error", f"{task_name} failed:\n{error}")
            if self._pending_auto_recommend:
                self.root.after(0, self._run_auto_recommendation_if_needed)
            return

        apply_result_fn(result)
        if self._pending_auto_recommend:
            self.root.after(0, self._run_auto_recommendation_if_needed)

    def _start_background_task(self, task_name, work_fn, apply_result_fn):
        if self._active_task_name:
            self._set_transient_status(f"{self._active_task_name} in progress...")
            return False

        self._active_task_id += 1
        task_id = self._active_task_id
        self._active_task_name = task_name
        self._pulse_active_task_status(task_id, task_name)

        def worker():
            result = None
            error = None
            error_trace = ""
            try:
                result = work_fn()
            except Exception as exc:
                error = exc
                error_trace = traceback.format_exc()
            self._task_result_queue.put((task_id, result, error, error_trace))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, lambda: self._poll_background_task(task_id, task_name, apply_result_fn))
        return True

    def _run_bp_recognition_from_path(self, image_path, source_label, candidate_top_k):
        image_path = Path(image_path)
        start_time = time.perf_counter()
        analysis = self._get_bp_analyzer().save_result(
            image_path,
            output_json_path=APP_BP_JSON_PATH,
            validate_monsters=True,
            candidate_top_k=candidate_top_k,
            validate_sides=("enemy",),
        )
        analysis_dict = analysis.to_dict()
        analysis_dict["my_slots"] = []
        return {
            "analysis": analysis_dict,
            "source": source_label,
            "image_path": str(image_path),
            "json_path": str(APP_BP_JSON_PATH),
            "candidate_top_k": candidate_top_k,
            "elapsed_seconds": time.perf_counter() - start_time,
        }

    def _apply_bp_recognition_result(self, result):
        analysis = result["analysis"]
        self.bp_recognition_result = result
        self.bp_recognition_source = result.get("source")
        self.bp_recognition_image_path = Path(result["image_path"]) if result.get("image_path") else None
        enemy_slots = analysis.get("enemy_slots", [])
        filled_count = sum(1 for slot in enemy_slots if slot.get("monster_name"))
        candidate_count = sum(1 for slot in enemy_slots if slot.get("monster_candidates"))
        elapsed_seconds = float(result.get("elapsed_seconds", 0.0))
        self.bp_recognition_summary_var.set(
            f"source={result.get('source', 'unknown')} | bp={analysis.get('is_bp_screen')} | "
            f"conf={float(analysis.get('confidence', 0.0)):.3f} | enemy={filled_count}/{len(enemy_slots)} | "
            f"candidate_slots={candidate_count} | top_n={int(result.get('candidate_top_k', 0) or 0)} | "
            f"{elapsed_seconds:.2f}s"
        )
        self.set_status("BP recognition finished")
        self.refresh_bp_state_panel()

    def recognize_bp_from_clipboard(self):
        try:
            clipboard_data = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read clipboard image: {exc}")
            return False

        if isinstance(clipboard_data, Image.Image):
            APP_BP_IMPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            image_to_save = clipboard_data
            if image_to_save.mode not in {"RGB", "RGBA", "L"}:
                image_to_save = image_to_save.convert("RGB")
            image_to_save.save(APP_BP_IMPORT_PATH)
            candidate_top_k = max(1, int(self.bp_enemy_candidate_top_n_var.get()))

            def work():
                return self._run_bp_recognition_from_path(APP_BP_IMPORT_PATH, "clipboard image", candidate_top_k)

            return self._start_background_task(
                "Recognizing clipboard screenshot", work, self._apply_bp_recognition_result
            )

        if isinstance(clipboard_data, list):
            image_candidates = [Path(item) for item in clipboard_data if is_supported_image_file(item)]
            if image_candidates:
                image_path = image_candidates[0]
                candidate_top_k = max(1, int(self.bp_enemy_candidate_top_n_var.get()))

                def work():
                    return self._run_bp_recognition_from_path(
                        image_path,
                        f"clipboard file: {image_path.name}",
                        candidate_top_k,
                    )

                return self._start_background_task(
                    "Recognizing clipboard screenshot", work, self._apply_bp_recognition_result
                )

        messagebox.showinfo(
            "Clipboard",
            "No image was found in the clipboard. Use your system screenshot tool first, then try again.",
        )
        return False

    def clear_bp_recognition(self):
        self.bp_recognition_result = None
        self.bp_recognition_source = None
        self.bp_recognition_image_path = None
        self.bp_recognition_summary_var.set("No BP recognition result yet")
        self.refresh_bp_state_panel()
        self.set_status("Cleared BP recognition result")

    def clear_children(self, widget):
        for child in widget.winfo_children():
            child.destroy()

    def refresh_all(self):
        self.set_phase()
        self.refresh_sidebar_pick_plans()
        self.refresh_bp_gallery()
        self.refresh_bp_state_panel()
        current_tab = self._current_tab_name()
        if self._box_tab_loaded or current_tab == "box":
            self.refresh_box_tab()
        if self._profile_tab_loaded or current_tab == "profile":
            self.refresh_profile_tab()

    def paginate(self, items, page, page_size):
        total_items = len(items)
        total_pages = max(1, math.ceil(total_items / max(1, page_size)))
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        return items[start:end], page, total_pages, total_items

    def _filter_monsters(
        self,
        search_text,
        attr_filter,
        family_filter,
        only_owned,
        only_with_image,
    ):
        monsters = list(self.monster_pool)
        if only_owned:
            owned = set(self.my_box_state)
            monsters = [m for m in monsters if m in owned]

        if search_text:
            search_text = search_text.strip().lower()
            monsters = [m for m in monsters if search_text in m.lower()]

        if attr_filter != "全部":
            monsters = [m for m in monsters if self.attr_map.get(m) == attr_filter]

        if family_filter != "全部":
            monsters = [m for m in monsters if self.family_map.get(m) == family_filter]

        if only_with_image:
            monsters = [m for m in monsters if get_image_path_for_monster(m, self.image_lookup)]

        return sort_monsters_by_pick_frequency(monsters, self.pick_freq)

    def render_image_card(
        self,
        parent,
        monster,
        size=(82, 82),
        subtitle="",
        primary_buttons=None,
        secondary_buttons=None,
        badge_text="",
        owned=None,
        card_command=None,
    ):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.RIDGE)
        if owned is not None:
            card.configure(highlightthickness=2, highlightbackground="#31a35c" if owned else "#d0d6e0")

        clickable_widgets = [card]
        if badge_text:
            badge = tk.Label(card, text=badge_text, bg="#e7eefc", fg="#3658a8", font=("", 8, "bold"))
            badge.pack(fill=tk.X)
            clickable_widgets.append(badge)

        photo = self.get_photo(monster, size=size)
        image_label = tk.Label(card, image=photo if photo else "", text="" if photo else "??", bg="#ffffff")
        image_label.image = photo
        image_label.pack(padx=5, pady=(5, 3))
        clickable_widgets.append(image_label)

        name_text = monster if len(monster) <= 10 else f"{monster[:9]}..."
        name_label = tk.Label(card, text=name_text, bg="#ffffff", font=("", 8, "bold"), wraplength=size[0] + 6, justify=tk.CENTER)
        name_label.pack(padx=4)
        clickable_widgets.append(name_label)

        if subtitle:
            subtitle_label = tk.Label(card, text=subtitle, bg="#ffffff", fg="#666666", font=("", 7), wraplength=size[0] + 6, justify=tk.CENTER)
            subtitle_label.pack(padx=4, pady=(2, 4))
            clickable_widgets.append(subtitle_label)
        else:
            spacer = tk.Label(card, text="", bg="#ffffff")
            spacer.pack(pady=(0, 3))
            clickable_widgets.append(spacer)

        if primary_buttons:
            row = tk.Frame(card, bg="#ffffff")
            row.pack(fill=tk.X, padx=4, pady=(0, 4))
            for text, command in primary_buttons:
                tk.Button(row, text=text, command=command, font=("", 8), height=1).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        if secondary_buttons:
            row = tk.Frame(card, bg="#ffffff")
            row.pack(fill=tk.X, padx=4, pady=(0, 6))
            for text, command in secondary_buttons:
                tk.Button(row, text=text, command=command, font=("", 8), height=1).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        if card_command:
            for widget in clickable_widgets:
                widget.bind("<Button-1>", lambda _event, cmd=card_command: cmd())
                try:
                    widget.configure(cursor="hand2")
                except Exception:
                    pass

        return card


    def refresh_bp_state_panel(self):
        self.bp_right_scroll.clear()
        parent = self.bp_right_scroll.inner

        # info_frame = ttk.LabelFrame(parent, text="Current BP", padding=10, style="Section.TLabelframe")
        # info_frame.pack(fill=tk.X, pady=(0, 10))
        # ttk.Label(info_frame, textvariable=self.phase_var, foreground="#3153a1", wraplength=300, justify=tk.LEFT).pack(anchor="w")
        # if self.last_action_var.get():
        #     ttk.Label(info_frame, textvariable=self.last_action_var, foreground="#2e7d32", wraplength=300, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))
        # ttk.Button(info_frame, text="Clear All", command=self.clear_all).pack(anchor="e", pady=(8, 0))

        self.render_pick_slots_section(parent, "My Picks", self.my_picks, "my")
        self.render_pick_slots_section(parent, "Enemy Picks", self.enemy_picks, "enemy")
        self.render_banned_section(parent)
        self.render_bp_recognition_section(parent)

    def render_pick_slots_section(self, parent, title, monsters, side):
        section = ttk.LabelFrame(parent, text=title, padding=8, style="Section.TLabelframe")
        section.pack(fill=tk.X, pady=(0, 10))
        row = ttk.Frame(section)
        row.pack(fill=tk.X)
        for idx in range(5):
            monster = monsters[idx] if idx < len(monsters) else None
            cell = tk.Frame(
                row,
                bg="#ffffff",
                bd=1,
                relief=tk.RIDGE,
                width=STATE_SLOT_CARD_SIZE["width"],
                height=STATE_SLOT_CARD_SIZE["height"],
            )
            cell.pack(side=tk.LEFT, padx=3, pady=3, expand=True, fill=tk.BOTH)
            cell.pack_propagate(False)
            if monster:
                photo = self.get_photo(monster, size=STATE_SLOT_IMAGE_SIZE)
                lbl = tk.Label(cell, image=photo if photo else "", text="" if photo else "No Img", bg="#ffffff")
                lbl.image = photo
                lbl.pack(pady=(5, 2))
                name_label = tk.Label(cell, text=monster, bg="#ffffff", wraplength=72, justify=tk.CENTER, font=("", 8, "bold"))
                name_label.pack(padx=3)
                hint_label = tk.Label(cell, text="点击取消", bg="#ffffff", fg="#7a8290", font=("", 7))
                hint_label.pack(pady=(5, 5))

                for widget in (cell, lbl, name_label, hint_label):
                    widget.bind("<Button-1>", lambda _event, m=monster: self.remove_one(m))
                    try:
                        widget.configure(cursor="hand2")
                    except Exception:
                        pass
            else:
                tk.Label(cell, text=f"{idx + 1}\nEmpty", bg="#f7f8fb", fg="#7a8290", justify=tk.CENTER, font=("", 10, "bold")).pack(expand=True, fill=tk.BOTH)








    def render_detected_enemy_bp_slots(self, parent, slots):
        block = ttk.LabelFrame(parent, text="Detected Enemy Candidates", padding=6, style="Section.TLabelframe")
        block.pack(fill=tk.X, pady=(6, 0))

        ordered_slots = sorted(
            slots,
            key=lambda slot: int(str(slot.get("slot_id", "0")).split("_")[-1]),
        )
        if not ordered_slots:
            return

        slot_row = ttk.Frame(block)
        slot_row.pack(anchor="w", fill=tk.X)

        candidate_limit = max(1, int(self.bp_enemy_candidate_top_n_var.get()))
        for col, slot in enumerate(ordered_slots):
            slot_frame = ttk.LabelFrame(
                slot_row,
                text=str(slot.get("slot_id", "enemy")),
                padding=6,
                style="Section.TLabelframe",
            )
            slot_frame.grid(row=0, column=col, padx=3, pady=(4, 0), sticky="nw")
            slot_row.grid_columnconfigure(col, weight=1)

            candidates = list(slot.get("monster_candidates") or [])[:candidate_limit]
            if not candidates:
                continue

            candidate_list = ttk.Frame(slot_frame)
            candidate_list.pack(anchor="w", pady=(6, 0))
            for idx, candidate in enumerate(candidates, start=1):
                monster = candidate.get("monster", "")
                subtitle = f"#{idx} score {float(candidate.get('score', 0.0)):.2f}"
                card = self.render_image_card(
                    candidate_list,
                    monster,
                    size=(52, 52),
                    subtitle=subtitle,
                    badge_text=f"#{idx}",
                    card_command=lambda m=monster: self.add_unique("enemy", m),
                )
                card.pack(anchor="w", pady=3)

    def render_bp_recognition_section(self, parent):
        section = ttk.LabelFrame(parent, text="Enemy BP Recognition", padding=8, style="Section.TLabelframe")
        section.pack(fill=tk.X, pady=(0, 10))

        toolbar_row = ttk.Frame(section)
        toolbar_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(toolbar_row, text="Paste Screenshot", command=self.recognize_bp_from_clipboard).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(toolbar_row, text="Clear Result", command=self.clear_bp_recognition).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Label(toolbar_row, text="Enemy top-N").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Spinbox(
            toolbar_row,
            from_=1,
            to=6,
            textvariable=self.bp_enemy_candidate_top_n_var,
            width=5,
        ).pack(side=tk.LEFT)

        # ttk.Label(
        #     section,
        #     textvariable=self.bp_recognition_summary_var,
        #     wraplength=320,
        #     justify=tk.LEFT,
        # ).pack(anchor="w")

        if not self.bp_recognition_result:
            return

        analysis = self.bp_recognition_result["analysis"]
        self.render_detected_enemy_bp_slots(section, analysis.get("enemy_slots", []))


    def refresh_box_tab(self):
        self._box_tab_loaded = True
        if not self._core_data_ready:
            self.box_count_label.configure(text=f"Box Count: {len(self.my_box_state)}")
            self.current_box_scroll.clear()
            ttk.Label(self.current_box_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            self.refresh_box_recommendations()
            self.box_gallery_scroll.clear()
            ttk.Label(self.box_gallery_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            return
        self.box_count_label.configure(text=f"Box Count: {len(self.my_box_state)}")

        current_box = sort_monsters_by_pick_frequency(self.my_box_state, self.pick_freq)
        self.current_box_scroll.clear()
        if not current_box:
            ttk.Label(self.current_box_scroll.inner, text="Box is empty").pack(anchor="w")
        else:
            for idx, monster in enumerate(current_box):
                row = idx // CURRENT_BOX_COLUMNS
                col = idx % CURRENT_BOX_COLUMNS
                card = self.render_image_card(
                    self.current_box_scroll.inner,
                    monster,
                    size=(66, 66),
                    primary_buttons=[("Remove", lambda m=monster: self.toggle_my_box(m))],
                    owned=True,
                )
                card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
                self.current_box_scroll.inner.grid_columnconfigure(col, weight=1)

        self.refresh_box_recommendations()

        monsters = self._filter_monsters(
            search_text=self.box_search_var.get(),
            attr_filter=self.box_attr_filter_var.get(),
            family_filter="全部",
            only_owned=self.box_only_owned_var.get(),
            only_with_image=self.box_only_with_image_var.get(),
        )
        page_size = max(20, int(self.box_page_size_var.get()))
        page_items, current_page, total_pages, total_items = self.paginate(monsters, self.box_gallery_page, page_size)
        self.box_gallery_page = current_page
        self.box_page_label.configure(text=f"Page {current_page}/{total_pages}, total {total_items}")
        self.box_prev_btn.configure(state=tk.NORMAL if current_page > 1 else tk.DISABLED)
        self.box_next_btn.configure(state=tk.NORMAL if current_page < total_pages else tk.DISABLED)

        self.box_gallery_scroll.clear()
        columns = BOX_GALLERY_COLUMNS
        owned_set = set(self.my_box_state)
        for idx, monster in enumerate(page_items):
            row = idx // columns
            col = idx % columns
            owned = monster in owned_set
            card = self.render_image_card(
                self.box_gallery_scroll.inner,
                monster,
                size=BOX_GALLERY_IMAGE_SIZE,
                subtitle=self.attr_map.get(monster, ""),
                primary_buttons=self._box_gallery_primary_buttons(monster, owned),
                owned=owned,
                badge_text="Owned" if owned else "",
            )
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self.box_gallery_scroll.inner.grid_columnconfigure(col, weight=1)

    def change_bp_page(self, delta):
        self.bp_gallery_page = max(1, self.bp_gallery_page + delta)
        self.refresh_bp_gallery()

    def change_box_page(self, delta):
        self.box_gallery_page = max(1, self.box_gallery_page + delta)
        self.refresh_box_tab()

    def remove_from_all(self, monster):
        if monster in self.my_picks:
            self.my_picks.remove(monster)
        if monster in self.enemy_picks:
            self.enemy_picks.remove(monster)
        if monster in self.banned_monsters:
            self.banned_monsters.remove(monster)
        if self.my_final_ban == monster:
            self.my_final_ban = None
        if self.enemy_final_ban == monster:
            self.enemy_final_ban = None
        if self.my_leader_var.get() == monster:
            self.my_leader_var.set("")
        if self.enemy_leader_var.get() == monster:
            self.enemy_leader_var.set("")










    def toggle_my_box(self, monster):
        if monster in self.my_box_state:
            self.my_box_state = [m for m in self.my_box_state if m != monster]
            self.set_status(f"已从我的 Box 移除 {monster}")
        else:
            self.my_box_state = sorted(set(self.my_box_state + [monster]))
            self.set_status(f"已加入我的 Box：{monster}")
        self.refresh_bp_gallery()
        self.refresh_box_tab()

    def save_my_box(self):
        self.my_box_state = sorted(set(self.my_box_state))
        save_json_file(MY_BOX_PATH, self.my_box_state)
        self.set_status(f"已保存我的 Box，共 {len(self.my_box_state)} 个魔灵")
        self.refresh_box_tab()

    def reload_my_box(self):
        self.my_box_state = sorted(set(load_json_file(MY_BOX_PATH, [])))
        self.set_status(f"已重新载入 my_box.json，共 {len(self.my_box_state)} 个魔灵")
        self.refresh_all()

    def clear_my_box(self):
        self.my_box_state = []
        self.set_status("已清空我的 Box")
        self.refresh_all()

    def build_candidate_pool(self):
        if not self._core_data_ready:
            return []
        occupied = set(self.my_picks) | set(self.enemy_picks) | set(self.banned_monsters)
        return [m for m in self.monster_pool if m not in occupied]




    def set_phase(self):
        my_count = len(self.my_picks)
        enemy_count = len(self.enemy_picks)

        if my_count == 5 and enemy_count == 5:
            if self.my_final_ban or self.enemy_final_ban:
                self.phase_var.set("已记录最终 Ban，BP 完成")
            else:
                self.phase_var.set("双方选人已完成，可以开始 Ban 推荐")
            return

        order, count = self._current_pick_span()
        side = self._current_pick_side()
        if order is None or side is None:
            self.phase_var.set("当前没有可用的选人位置")
            return

        side_label = "我方" if side == "my" else "敌方"
        if count and count > 1:
            self.phase_var.set(f"当前为{side_label}双选阶段（第 {order}-{order + count - 1} 手）")
        else:
            self.phase_var.set(f"当前为{side_label}选人阶段（第 {order} 手）")

    def build_sidebar_prediction_card(self, parent, prediction, idx):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.RIDGE)
        title_label = tk.Label(card, text=f"Pred {idx}", bg="#ffffff", font=("", 9, "bold"))
        title_label.pack(anchor="w", padx=6, pady=(5, 3))

        monster = prediction.get("monster", "")
        photo = self.get_photo(monster, size=SIDEBAR_RESULT_IMAGE_SIZE)
        image_label = tk.Label(card, image=photo if photo else "", text="" if photo else "No Img", bg="#ffffff")
        image_label.image = photo
        image_label.pack(pady=(0, 3))
        short_name = monster if len(monster) <= 8 else f"{monster[:7]}..."
        name_label = tk.Label(
            card,
            text=short_name,
            bg="#ffffff",
            font=("", 8, "bold"),
            wraplength=82,
            justify=tk.CENTER,
        )
        name_label.pack(padx=6)

        info = tk.Frame(card, bg="#ffffff")
        info.pack(fill=tk.X, padx=6, pady=(4, 4))
        info_label = tk.Label(info, text=f"Score {float(prediction.get('final_score', 0.0)):.4f}", bg="#ffffff", anchor="w", font=("", 8))
        info_label.pack(fill=tk.X)

        for widget in (card, title_label, image_label, name_label, info, info_label):
            widget.bind("<Button-1>", lambda _event, m=monster: self.add_to_current_turn(m))
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass
        return card

    def refresh_bp_gallery(self):
        if not self._core_data_ready:
            self.bp_gallery_count_label.configure(text="Loading...")
            self.bp_page_label.configure(text="Loading")
            self.bp_prev_btn.configure(state=tk.DISABLED)
            self.bp_next_btn.configure(state=tk.DISABLED)
            self.bp_gallery_scroll.clear()
            ttk.Label(self.bp_gallery_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            return

        monsters = self._filter_monsters(
            search_text=self.search_var.get(),
            attr_filter=self.attr_filter_var.get(),
            family_filter=self.family_filter_var.get(),
            only_owned=self.only_owned_in_picker_var.get(),
            only_with_image=self.only_with_image_var.get(),
        )

        page_size = self._effective_bp_page_size()
        page_items, current_page, total_pages, total_items = self.paginate(monsters, self.bp_gallery_page, page_size)
        self.bp_gallery_page = current_page
        self.bp_gallery_count_label.configure(text=f"Count: {total_items}")
        self.bp_page_label.configure(text=f"Page {current_page}/{total_pages}")
        self.bp_prev_btn.configure(state=tk.NORMAL if current_page > 1 else tk.DISABLED)
        self.bp_next_btn.configure(state=tk.NORMAL if current_page < total_pages else tk.DISABLED)

        self.bp_gallery_scroll.clear()
        columns = BP_GALLERY_COLUMNS
        for idx, monster in enumerate(page_items):
            row = idx // columns
            col = idx % columns
            card = self.render_image_card(
                self.bp_gallery_scroll.inner,
                monster,
                size=BP_GALLERY_IMAGE_SIZE,
                subtitle=self.attr_map.get(monster, ""),
                card_command=lambda m=monster: self.add_to_current_turn(m),
            )
            card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            self.bp_gallery_scroll.inner.grid_columnconfigure(col, weight=1)


    def add_to_current_turn(self, monster):
        order = self._current_pick_order()
        side = self._current_pick_side()
        if order is None or side is None:
            messagebox.showinfo("提示", "双方已经完成选人，当前不能继续填入魔灵。")
            return

        side_label = "我方" if side == "my" else "敌方"
        self.add_unique(side, monster, status_text=f"已将 {monster} 填入{side_label}第 {order} 手")

    def add_unique(self, side, monster, status_text=None):
        self.remove_from_all(monster)
        target = self.my_picks if side == "my" else self.enemy_picks
        if len(target) >= 5:
            messagebox.showwarning("提示", f"{'我方' if side == 'my' else '敌方'}已满，无法继续加入。")
            return

        target.append(monster)
        final_status = status_text or f"已将 {monster} 加入 {'我方' if side == 'my' else '敌方'}"
        self._apply_bp_state_change(final_status)

    def add_preban(self, monster):
        self.remove_from_all(monster)
        if monster not in self.banned_monsters:
            self.banned_monsters.append(monster)
        self._apply_bp_state_change(f"已将 {monster} 加入预禁")

    def remove_one(self, monster):
        self.remove_from_all(monster)
        self._apply_bp_state_change(f"已移除 {monster}")

    def clear_all(self):
        self.my_picks = []
        self.enemy_picks = []
        self.banned_monsters = []
        self.my_final_ban = None
        self.enemy_final_ban = None
        self.my_leader_var.set("")
        self.enemy_leader_var.set("")
        self.enemy_ban_candidate_var.set("")
        self._apply_bp_state_change("已清空当前 BP 状态")

    def apply_pick_plan(self, plan):
        targets = [x for x in plan.get("plan", []) if x]
        remain = 5 - len(self.my_picks)
        unique_targets = []
        for monster in targets:
            if monster not in unique_targets:
                unique_targets.append(monster)

        to_add = [m for m in unique_targets if m not in self.my_picks]
        if len(to_add) > remain:
            messagebox.showwarning("提示", f"我方剩余位置不足，仅剩 {remain} 个。")
            return

        for monster in to_add:
            self.remove_from_all(monster)
            self.my_picks.append(monster)

        joined = " + ".join(to_add) if to_add else "当前方案"
        self._apply_bp_state_change(f"已采用推荐方案：{joined}")

    def apply_my_final_ban(self, monster):
        self.my_final_ban = monster
        self._apply_bp_state_change(f"已采用我方 Ban：{monster}")

    def record_enemy_ban(self):
        candidate = self.enemy_ban_candidate_var.get().strip()
        if not candidate:
            messagebox.showinfo("提示", "请先选择敌方 Ban 目标。")
            return

        self.enemy_final_ban = candidate
        self._apply_bp_state_change(f"已记录对方 Ban：{candidate}")

    def clear_final_bans(self):
        self.my_final_ban = None
        self.enemy_final_ban = None
        self.enemy_ban_candidate_var.set("")
        self._apply_bp_state_change("已清空最终 Ban")

    def generate_pick_recommendations(self, silent=False):
        if not self._ensure_core_data_ready(silent=silent):
            return False
        turn = infer_turn_from_state(self.my_picks, self.enemy_picks, self.first_pick_var.get())
        stage = infer_bp_stage(self.my_picks, self.enemy_picks)
        if stage not in {"pick_single", "pick_multi"} or turn is None:
            if not silent:
                messagebox.showinfo("Notice", "It is not my pick turn right now.")
            return False

        pick_count = 1 if stage == "pick_single" else 2
        state_snapshot = {
            "my_picks": list(self.my_picks),
            "enemy_picks": list(self.enemy_picks),
            "turn": int(turn),
            "stage": stage,
            "first_pick_side": self.first_pick_var.get(),
            "top_k": max(1, int(self.top_k_var.get())),
            "candidate_pool": list(self.build_candidate_pool()),
            "my_box": list(self.my_box_state),
            "use_my_box_for_reco": bool(self.use_my_box_for_reco_var.get()),
            "my_leader": self.my_leader_var.get().strip(),
            "enemy_leader": self.enemy_leader_var.get().strip(),
            "my_final_ban": self.my_final_ban,
            "enemy_final_ban": self.enemy_final_ban,
            "pick_count": pick_count,
            "context_signature": self._recommendation_context_signature(),
        }

        def work():
            relation_recommender = self._get_relation_recommender()
            plans = relation_recommender.recommend_pick_plans(
                my_picks=state_snapshot["my_picks"],
                enemy_picks=state_snapshot["enemy_picks"],
                pick_count=state_snapshot["pick_count"],
                turn=state_snapshot["turn"],
                first_pick_side=state_snapshot["first_pick_side"],
                top_k=state_snapshot["top_k"],
                candidate_pool=state_snapshot["candidate_pool"],
                my_box=state_snapshot["my_box"],
                use_my_box_for_reco=state_snapshot["use_my_box_for_reco"],
            )
            return self._attach_pick_plan_win_rates(plans, state_snapshot)

        def apply_result(plans):
            if state_snapshot["context_signature"] != self._recommendation_context_signature():
                self.set_status("Pick recommendations were discarded because the BP state changed")
                return
            self.pick_plans = plans
            self.set_status(f"Generated {len(plans)} pick plans")
            self.refresh_sidebar_pick_plans()
            self.refresh_bp_state_panel()

        return self._start_background_task("Generating pick recommendations", work, apply_result)

    def _attach_pick_plan_win_rates(self, plans, state_snapshot):
        if not plans:
            return plans

        try:
            ban_value_recommender = self._get_ban_value_recommender()
            base_eval = ban_value_recommender.evaluate_state(
                my_picks=state_snapshot["my_picks"],
                enemy_picks=state_snapshot["enemy_picks"],
                my_ban=state_snapshot["my_final_ban"],
                enemy_ban=state_snapshot["enemy_final_ban"],
                my_leader=state_snapshot["my_leader"],
                enemy_leader=state_snapshot["enemy_leader"],
                return_detail=True,
            )
            base_wr = float(base_eval.get("calibrated_win_rate", 0.0))
            base_raw_wr = float(base_eval.get("raw_win_rate", 0.0))
        except Exception as exc:
            print(f"[WARN] failed to evaluate base pick win rate: {exc}")
            base_wr = 0.0
            base_raw_wr = 0.0

        enriched = []
        for plan in plans:
            item = dict(plan)
            plan_picks = [monster for monster in item.get("plan", []) if monster]
            future_my_picks = list(state_snapshot["my_picks"]) + plan_picks
            detail = dict(item.get("detail", {}) or {})
            try:
                eval_result = ban_value_recommender.evaluate_state(
                    my_picks=future_my_picks,
                    enemy_picks=state_snapshot["enemy_picks"],
                    my_ban=state_snapshot["my_final_ban"],
                    enemy_ban=state_snapshot["enemy_final_ban"],
                    my_leader=state_snapshot["my_leader"],
                    enemy_leader=state_snapshot["enemy_leader"],
                    return_detail=True,
                )
                wr = float(eval_result.get("calibrated_win_rate", 0.0))
                raw_wr = float(eval_result.get("raw_win_rate", 0.0))
                item["estimated_win_rate"] = round(wr, 6)
                detail.update(
                    {
                        "pre_pick_win_rate": round(base_wr, 6),
                        "post_pick_win_rate": round(wr, 6),
                        "win_rate_gain": round(wr - base_wr, 6),
                        "raw_pre_pick_win_rate": round(base_raw_wr, 6),
                        "raw_post_pick_win_rate": round(raw_wr, 6),
                        "raw_win_rate_gain": round(raw_wr - base_raw_wr, 6),
                        "win_rate_source": "draft_value_display_only",
                    }
                )
            except Exception as exc:
                print(f"[WARN] failed to evaluate pick plan win rate for {plan_picks}: {exc}")
                item["estimated_win_rate"] = 0.0
                detail.setdefault("win_rate_source", "unavailable")
            item["detail"] = detail
            enriched.append(item)
        return enriched

    def generate_enemy_predictions(self, silent=False):
        if not self._ensure_core_data_ready(silent=silent):
            return False
        stage = infer_bp_stage(self.my_picks, self.enemy_picks)
        current_side = self._current_pick_side()
        if stage not in {"pick_single", "pick_multi"} or current_side != "enemy":
            if not silent:
                messagebox.showinfo("Notice", "It is not the enemy pick turn right now.")
            return False

        state_snapshot = {
            "my_picks": list(self.my_picks),
            "enemy_picks": list(self.enemy_picks),
            "first_pick_side": self.first_pick_var.get(),
            "top_k": max(1, int(self.top_k_var.get())),
            "candidate_pool": list(self.build_candidate_pool()),
            "my_leader": self.my_leader_var.get().strip(),
            "enemy_leader": self.enemy_leader_var.get().strip(),
            "context_signature": self._recommendation_context_signature(),
        }

        def work():
            relation_recommender = self._get_relation_recommender()
            return relation_recommender.recommend_enemy_next_picks(
                my_picks=state_snapshot["my_picks"],
                enemy_picks=state_snapshot["enemy_picks"],
                first_pick_side=state_snapshot["first_pick_side"],
                top_k=state_snapshot["top_k"],
                candidate_pool=state_snapshot["candidate_pool"],
                my_leader=state_snapshot["my_leader"],
                enemy_leader=state_snapshot["enemy_leader"],
            )

        def apply_result(predictions):
            if state_snapshot["context_signature"] != self._recommendation_context_signature():
                self.set_status("Enemy predictions were discarded because the BP state changed")
                return
            self.enemy_predictions = predictions
            self.set_status(f"Generated {len(predictions)} enemy predictions")
            self.refresh_sidebar_pick_plans()
            self.refresh_bp_state_panel()

        return self._start_background_task("Generating enemy predictions", work, apply_result)

    def generate_ban_recommendations(self, silent=False):
        if not self._ensure_core_data_ready(silent=silent):
            return False
        if len(self.my_picks) != 5 or len(self.enemy_picks) != 5:
            if not silent:
                messagebox.showinfo("Notice", "Ban recommendations need both sides to finish 5 picks.")
            return False

        state_snapshot = {
            "my_picks": list(self.my_picks),
            "enemy_picks": list(self.enemy_picks),
            "my_leader": self.my_leader_var.get().strip(),
            "enemy_leader": self.enemy_leader_var.get().strip(),
            "enemy_ban": self.enemy_final_ban,
            "top_k": max(1, int(self.top_k_var.get())),
            "context_signature": self._recommendation_context_signature(),
        }

        def work():
            ban_value_recommender = self._get_ban_value_recommender()
            try:
                relation_recommender = self._get_relation_recommender()
                return relation_recommender.recommend_ban_plans(
                    my_picks=state_snapshot["my_picks"],
                    enemy_picks=state_snapshot["enemy_picks"],
                    top_k=state_snapshot["top_k"],
                    old_ban_recommender=ban_value_recommender,
                )
            except Exception as exc:
                print(f"[WARN] relation ban recommender failed, fallback to value ban: {exc}")
                return ban_value_recommender.recommend_bans(
                    my_picks=state_snapshot["my_picks"],
                    enemy_picks=state_snapshot["enemy_picks"],
                    my_leader=state_snapshot["my_leader"],
                    enemy_leader=state_snapshot["enemy_leader"],
                    enemy_ban=state_snapshot["enemy_ban"],
                    top_k=state_snapshot["top_k"],
                )

        def apply_result(plans):
            if state_snapshot["context_signature"] != self._recommendation_context_signature():
                self.set_status("Ban recommendations were discarded because the BP state changed")
                return
            self.ban_plans = plans
            self.set_status(f"Generated {len(plans)} ban plans")
            self.refresh_sidebar_pick_plans()
            self.refresh_bp_state_panel()

        return self._start_background_task("Generating ban recommendations", work, apply_result)

    def _build_main_tabs(self):
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.bp_tab = ttk.Frame(self.notebook)
        self.profile_tab = ttk.Frame(self.notebook)
        self.box_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bp_tab, text="BP 模式")
        self.notebook.add(self.profile_tab, text="Box 推荐")
        self.notebook.add(self.box_tab, text="我的 Box 管理")

        self.notebook.hide(self.profile_tab)

        self._build_bp_tab()
        self._build_box_tab()

    def _bind_var_events(self):
        self.first_pick_var.trace_add("write", self.on_first_pick_side_changed)

        for variable in [
            self.search_var,
            self.attr_filter_var,
            self.family_filter_var,
            self.bp_page_size_var,
        ]:
            variable.trace_add("write", lambda *_: self.on_bp_filter_changed())

        for variable in [
            self.box_search_var,
            self.box_attr_filter_var,
            self.box_page_size_var,
        ]:
            variable.trace_add("write", lambda *_: self.on_box_filter_changed())

        self.bp_enemy_candidate_top_n_var.trace_add("write", lambda *_: self.refresh_bp_state_panel())

    def _build_profile_tab(self):
        for child in self.profile_tab.winfo_children():
            child.destroy()
        self._build_box_recommendation_tab()

    def _build_box_recommendation_tab(self):
        layout = ttk.Frame(self.profile_tab)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=4)
        layout.columnconfigure(1, weight=7)
        layout.rowconfigure(0, weight=1)

        control = ttk.Frame(layout, padding=(0, 0, 10, 0))
        control.grid(row=0, column=0, sticky="nsew")
        result = ttk.Frame(layout)
        result.grid(row=0, column=1, sticky="nsew")

        controls = ttk.LabelFrame(control, text="Box 推荐控制", padding=10, style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(0, 8))

        season_row = ttk.Frame(controls)
        season_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(season_row, text="样本范围", width=10).pack(side=tk.LEFT)
        self.comp_season_combo = ttk.Combobox(
            season_row,
            textvariable=self.comp_reco_season_var,
            values=["全部样本"],
            state="readonly",
            width=18,
        )
        self.comp_season_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mode_row = ttk.Frame(controls)
        mode_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(mode_row, text="推荐类型", width=10).pack(side=tk.LEFT)
        self.comp_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.comp_reco_mode_var,
            values=[
                "当前赛季最强 Box",
                "当前赛季三系最强 Box",
                "基于核心构建 Box",
            ],
            state="readonly",
            width=24,
        )
        self.comp_mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        size_row = ttk.Frame(controls)
        size_row.pack(fill=tk.X)
        ttk.Label(size_row, text="方案数量", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(size_row, from_=3, to=12, textvariable=self.comp_reco_top_k_var, width=6).pack(side=tk.LEFT)
        ttk.Label(size_row, text="Box 大小", width=10).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Spinbox(size_row, from_=8, to=12, textvariable=self.comp_box_size_var, width=6).pack(side=tk.LEFT)

        core_frame = ttk.LabelFrame(control, text="核心魔灵", padding=10, style="Section.TLabelframe")
        core_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        search_row = ttk.Frame(core_frame)
        search_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(search_row, text="搜索", width=8).pack(side=tk.LEFT)
        ttk.Entry(search_row, textvariable=self.comp_core_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(search_row, text="加入核心", command=self.add_selected_core_monster).pack(side=tk.RIGHT)

        self.comp_core_listbox = tk.Listbox(core_frame, height=10, exportselection=False)
        self.comp_core_listbox.pack(fill=tk.BOTH, expand=True)
        self.comp_core_listbox.bind("<Double-Button-1>", lambda _event: self.add_selected_core_monster())

        selected_header = ttk.Frame(core_frame)
        selected_header.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(selected_header, text="已选核心", font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(selected_header, text="清空核心", command=self.clear_core_monsters).pack(side=tk.RIGHT)

        self.comp_core_selected_frame = ttk.Frame(core_frame)
        self.comp_core_selected_frame.pack(fill=tk.X)

        self.comp_generate_btn = ttk.Button(control, text="生成 Box 推荐", command=self.generate_comp_recommendations)
        self.comp_generate_btn.pack(fill=tk.X, pady=(2, 0))

        result_top = ttk.Frame(result)
        result_top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(result_top, text="Box 推荐结果", font=("", 12, "bold")).pack(side=tk.LEFT)
        self.comp_result_count_label = ttk.Label(result_top, text="", foreground="#666666")
        self.comp_result_count_label.pack(side=tk.RIGHT)

        self.comp_results_scroll = ScrollableFrame(result)
        self.comp_results_scroll.pack(fill=tk.BOTH, expand=True)

        self.comp_reco_mode_var.set("当前赛季最强 Box")
        self.refresh_profile_tab()

    def _profile_available_seasons(self):
        seasons = []
        for sample in self.samples or []:
            season = str(sample.get("season") or "").strip()
            if season:
                seasons.append(season)
        seasons = sorted(set(seasons))
        return ["全部样本"] + seasons if seasons else ["全部样本"]

    def _refresh_comp_season_options(self):
        if not hasattr(self, "comp_season_combo"):
            return
        options = self._profile_available_seasons() if self._core_data_ready else ["全部样本"]
        self.comp_season_combo.configure(values=options)
        if self.comp_reco_season_var.get() not in options:
            self.comp_reco_season_var.set(options[0])

    def _current_comp_season(self):
        value = self.comp_reco_season_var.get().strip()
        return None if not value or value == "全部样本" else value

    def _iter_box_match_records(self, season=None):
        seen = set()
        season_value = str(season or "").strip()
        for sample in self.samples or []:
            sample_season = str(sample.get("season") or "").strip()
            if season_value and sample_season != season_value:
                continue
            final_my = [monster for monster in sample.get("final_my_picks", []) if monster]
            if len(final_my) != 5:
                continue
            match_id = str(sample.get("match_id") or "").strip()
            if not match_id:
                match_id = "|".join(final_my) + "|" + str(sample.get("source_html") or "")
            if match_id in seen:
                continue
            seen.add(match_id)
            won = sample.get("won")
            if won is None:
                text = str(sample.get("result") or "").strip().lower()
                if text in {"win", "victory", "winner"}:
                    won = True
                elif text in {"loss", "lose", "defeat", "loser"}:
                    won = False
            yield {
                "match_id": match_id,
                "box": list(dict.fromkeys(final_my)),
                "won": won,
            }

    def _build_box_meta_stats(self, season=None):
        monster_games = Counter()
        monster_wins = Counter()
        pair_games = Counter()
        pair_wins = Counter()
        records = list(self._iter_box_match_records(season=season))
        for record in records:
            box = record["box"]
            won = bool(record.get("won"))
            for monster in box:
                monster_games[monster] += 1
                if won:
                    monster_wins[monster] += 1
            for a, b in combinations(sorted(box), 2):
                pair_games[(a, b)] += 1
                if won:
                    pair_wins[(a, b)] += 1
        return {
            "records": records,
            "monster_games": monster_games,
            "monster_wins": monster_wins,
            "pair_games": pair_games,
            "pair_wins": pair_wins,
            "global_win_rate": self._box_global_win_rate(records),
            "max_monster_games": max(monster_games.values()) if monster_games else 1,
            "max_pair_games": max(pair_games.values()) if pair_games else 1,
        }

    def _box_global_win_rate(self, records):
        wins = 0
        games = 0
        for record in records:
            if record.get("won") is None:
                continue
            games += 1
            if record.get("won"):
                wins += 1
        return (wins + 4.0) / (games + 8.0) if games > 0 else 0.5

    def _smoothed_rate(self, wins, games, prior_win_rate=0.5, prior_count=8.0):
        return (float(wins) + prior_win_rate * prior_count) / (float(games) + prior_count)

    def _monster_meta_score(self, monster, stats):
        games = float(stats["monster_games"].get(monster, 0))
        wins = float(stats["monster_wins"].get(monster, 0))
        smoothed_wr = self._smoothed_rate(wins, games)
        support_score = math.log1p(games) / math.log1p(float(stats["max_monster_games"]) + 1.0)
        pick_score = math.log1p(float(self.pick_freq.get(monster, 0))) / math.log1p(float(max(self.pick_freq.values()) if self.pick_freq else 1) + 1.0)
        return 0.55 * smoothed_wr + 0.30 * support_score + 0.15 * pick_score

    def _pair_meta_score(self, a, b, stats):
        key = tuple(sorted((a, b)))
        games = float(stats["pair_games"].get(key, 0))
        wins = float(stats["pair_wins"].get(key, 0))
        if games <= 0:
            return 0.5
        smoothed_wr = self._smoothed_rate(wins, games)
        support_score = math.log1p(games) / math.log1p(float(stats["max_pair_games"]) + 1.0)
        return 0.60 * smoothed_wr + 0.40 * support_score

    def _box_candidate_pool(self, stats, tri_only=False):
        candidates = list(stats["monster_games"].keys()) or list(self.monster_pool)
        if tri_only:
            candidates = [
                monster
                for monster in candidates
                if str(self.attr_map.get(monster, "")).strip().lower() in {"fire", "water", "wind"}
            ]
        ranked = sorted(
            dict.fromkeys(candidates),
            key=lambda monster: (self._monster_meta_score(monster, stats), self.pick_freq.get(monster, 0), monster),
            reverse=True,
        )
        return ranked

    def _box_profile_bonus(self, selected, candidate):
        if self.recommender is None:
            return 0.0, []
        try:
            profile = self.recommender._analyze_comp_profile(selected[:5] if len(selected) >= 5 else selected)
            need_score, need_reasons = self.recommender._my_need_score(profile, candidate)
            syn_score, syn_reasons = self.recommender._synergy_score(profile, candidate)
            return 0.18 * need_score + 0.12 * syn_score, list(need_reasons[:1]) + list(syn_reasons[:1])
        except Exception:
            return 0.0, []

    def _greedy_build_box(self, seed_monsters, stats, box_size=10, tri_only=False):
        selected = list(dict.fromkeys([monster for monster in seed_monsters if monster]))
        candidate_pool = self._box_candidate_pool(stats, tri_only=tri_only)
        while len(selected) < box_size:
            current_attrs = {str(self.attr_map.get(monster, "")).strip().lower() for monster in selected if self.attr_map.get(monster)}
            best_monster = None
            best_score = -1e9
            for candidate in candidate_pool:
                if candidate in selected:
                    continue
                single_score = self._monster_meta_score(candidate, stats)
                pair_values = [self._pair_meta_score(candidate, other, stats) for other in selected] if selected else [0.5]
                pair_score = sum(pair_values) / max(1, len(pair_values))
                profile_bonus, _reasons = self._box_profile_bonus(selected, candidate)
                attr_bonus = 0.0
                if tri_only:
                    attr = str(self.attr_map.get(candidate, "")).strip().lower()
                    if attr in {"fire", "water", "wind"} and attr not in current_attrs:
                        attr_bonus = 0.06
                total = 0.52 * single_score + 0.38 * pair_score + profile_bonus + attr_bonus
                if total > best_score:
                    best_score = total
                    best_monster = candidate
            if not best_monster:
                break
            selected.append(best_monster)
        return selected[:box_size]

    def _score_box_package(self, box_monsters, stats, core_monsters=None):
        core_monsters = core_monsters or []
        if not box_monsters:
            return 0.0
        single = [self._monster_meta_score(monster, stats) for monster in box_monsters]
        pair_scores = [self._pair_meta_score(a, b, stats) for a, b in combinations(box_monsters, 2)]
        avg_single = sum(single) / len(single)
        avg_pair = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
        core_fit = 1.0
        if core_monsters:
            core_fit = len([monster for monster in core_monsters if monster in box_monsters]) / max(1, len(core_monsters))
        return 0.48 * avg_single + 0.36 * avg_pair + 0.16 * core_fit

    def _infer_box_style(self, monsters):
        if self.recommender is None or not monsters:
            return "balanced", "平衡 Box"
        try:
            profile = self.recommender._analyze_comp_profile(monsters[:5])
            speed = float(profile.get("aggressive_speed_score", 0.0))
            protect = float(profile.get("protective_score", 0.0))
            anti = float(profile.get("anti_control_score", 0.0))
            damage = float(profile.get("damage_pressure_score", 0.0))
            if speed >= protect and speed >= anti and speed >= damage:
                return "speed_control", "速控 Box"
            if protect >= speed and protect >= anti:
                return "bruiser", "肉队 Box"
            if anti >= speed and anti >= protect:
                return "anti_cc", "反控 Box"
            if damage >= speed and damage >= protect:
                return "burst", "爆发 Box"
        except Exception:
            pass
        return "balanced", "平衡 Box"

    def _build_box_recommendation_item(self, box_monsters, stats, mode, core_monsters=None):
        core_monsters = core_monsters or []
        score = self._score_box_package(box_monsters, stats, core_monsters=core_monsters)
        style_code, style_name = self._infer_box_style(box_monsters)
        supports = [int(stats["monster_games"].get(monster, 0)) for monster in box_monsters]
        box_support = round(sum(supports) / max(1, len(supports)), 2)
        core_included = [monster for monster in core_monsters if monster in box_monsters]
        reasons = []
        if mode == "当前赛季最强 Box":
            reasons.append("按赛季样本中的单卡强度、共现协同和稳定出场率综合排序。")
        elif mode == "当前赛季三系最强 Box":
            reasons.append("仅从火 / 水 / 风魔灵中筛选，优先保留三系可用性和协同。")
        else:
            if core_included:
                reasons.append("围绕所选核心扩展，优先保留核心的强协同与常见后续拼法。")
            else:
                reasons.append("核心样本不足时，退化为通用高强度 Box 方案。")
        if core_included:
            reasons.append("已包含核心：" + ", ".join(core_included[:5]))
        reasons.append(f"平均支持度 {box_support:.2f}，兼顾稳定性而不是只看高频。")
        return {
            "box": box_monsters,
            "score": round(float(score), 6),
            "name": style_name,
            "style": style_code,
            "reasons": reasons,
            "detail": {
                "avg_support": box_support,
                "core_match_count": len(core_included),
                "box_size": len(box_monsters),
            },
        }

    def _box_jaccard(self, a, b):
        sa = {normalize_name(monster) for monster in a if normalize_name(monster)}
        sb = {normalize_name(monster) for monster in b if normalize_name(monster)}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _generate_box_recommendations(self, mode, season=None, core_monsters=None, top_k=6, box_size=10):
        core_monsters = list(dict.fromkeys(core_monsters or []))
        stats = self._build_box_meta_stats(season=season)
        tri_only = mode == "当前赛季三系最强 Box"
        if not stats["records"]:
            return []

        pool = self._box_candidate_pool(stats, tri_only=tri_only)
        seeds = []
        if mode == "基于核心构建 Box" and core_monsters:
            seeds.append(core_monsters[:])
            for monster in pool[:8]:
                if monster not in core_monsters:
                    seeds.append(core_monsters + [monster])
        else:
            for monster in pool[: min(10, len(pool))]:
                seeds.append([monster])
            for a, b in combinations(pool[: min(6, len(pool))], 2):
                seeds.append([a, b])

        raw_items = []
        seen_boxes = set()
        for seed in seeds:
            box_monsters = self._greedy_build_box(seed, stats, box_size=max(8, min(12, int(box_size))), tri_only=tri_only)
            if core_monsters and any(monster not in box_monsters for monster in core_monsters):
                continue
            if len(box_monsters) < max(8, min(12, int(box_size))):
                continue
            key = tuple(box_monsters)
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
            raw_items.append(self._build_box_recommendation_item(box_monsters, stats, mode, core_monsters=core_monsters))

        raw_items.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        selected = []
        for item in raw_items:
            if any(self._box_jaccard(item["box"], keep["box"]) > 0.82 for keep in selected):
                continue
            selected.append(item)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def refresh_comp_core_candidates(self):
        if not hasattr(self, "comp_core_listbox"):
            return
        self.comp_core_listbox.delete(0, tk.END)
        if not self._core_data_ready:
            self.comp_core_listbox.insert(tk.END, "Core data is loading...")
            return
        search_text = self.comp_core_search_var.get().strip().lower()
        selected = set(self.comp_core_monsters)
        monsters = sort_monsters_by_pick_frequency(list(self.monster_pool), self.pick_freq)
        if search_text:
            monsters = [monster for monster in monsters if search_text in monster.lower()]
        for monster in monsters[:250]:
            if monster not in selected:
                self.comp_core_listbox.insert(tk.END, monster)

    def refresh_selected_core_monsters(self):
        if not hasattr(self, "comp_core_selected_frame"):
            return
        self.clear_children(self.comp_core_selected_frame)
        if not self.comp_core_monsters:
            ttk.Label(
                self.comp_core_selected_frame,
                text="不参考当前已有 Box。这里的核心只用于指定你想围绕哪些魔灵构建一整套 Box。",
                foreground="#666666",
                wraplength=320,
                justify=tk.LEFT,
            ).pack(anchor="w")
            return
        columns = 3
        for idx, monster in enumerate(self.comp_core_monsters):
            card = self.render_image_card(
                self.comp_core_selected_frame,
                monster,
                size=(64, 64),
                subtitle=self.attr_map.get(monster, ""),
                primary_buttons=[("移除", lambda m=monster: self.remove_core_monster(m))],
                owned=monster in set(self.my_box_state),
            )
            row = idx // columns
            col = idx % columns
            card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            self.comp_core_selected_frame.grid_columnconfigure(col, weight=1)

    def add_selected_core_monster(self):
        if not hasattr(self, "comp_core_listbox"):
            return
        selection = self.comp_core_listbox.curselection()
        if not selection:
            return
        self.add_core_monster(self.comp_core_listbox.get(selection[0]))

    def add_core_monster(self, monster):
        monster = str(monster or "").strip()
        if not monster or monster in self.comp_core_monsters:
            return
        if len(self.comp_core_monsters) >= 5:
            self.set_status("Core limit is 5")
            return
        self.comp_core_monsters.append(monster)
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def remove_core_monster(self, monster):
        self.comp_core_monsters = [item for item in self.comp_core_monsters if item != monster]
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def clear_core_monsters(self):
        self.comp_core_monsters = []
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def set_core_monsters_from_box(self, monsters):
        self.comp_core_monsters = list(monsters[:5])
        self.comp_reco_mode_var.set("基于核心构建 Box")
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def _invalidate_comp_recommendations(self, *_args):
        self.comp_recommendations = []
        self.refresh_profile_tab()

    def refresh_profile_tab(self):
        self._profile_tab_loaded = True
        self._refresh_comp_season_options()
        self.refresh_comp_core_candidates()
        self.refresh_selected_core_monsters()
        self.refresh_comp_recommendation_results()
        if hasattr(self, "comp_generate_btn"):
            self.comp_generate_btn.configure(state=tk.NORMAL if self._core_data_ready else tk.DISABLED)

    def on_profile_filter_changed(self):
        self.comp_recommendations = []
        if self._profile_tab_loaded or self._current_tab_name() == "profile":
            self.refresh_profile_tab()

    def generate_comp_recommendations(self):
        if not self._ensure_core_data_ready():
            return
        mode = self.comp_reco_mode_var.get().strip() or "当前赛季最强 Box"
        top_k = max(1, int(self.comp_reco_top_k_var.get() or 6))
        box_size = max(8, min(12, int(self.comp_box_size_var.get() or 10)))
        season = self._current_comp_season()
        cores = list(self.comp_core_monsters)

        def work():
            return self._generate_box_recommendations(
                mode=mode,
                season=season,
                core_monsters=cores,
                top_k=top_k,
                box_size=box_size,
            )

        def apply_result(plans):
            self.comp_recommendations = list(plans or [])
            self.refresh_comp_recommendation_results()
            self.set_status(f"Generated {len(self.comp_recommendations)} box recommendations")

        return self._start_background_task("Generating box recommendations", work, apply_result)

    def copy_comp_to_clipboard(self, comp):
        text = ", ".join(comp or [])
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.set_status(f"Copied box: {text}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to copy box: {exc}")

    def refresh_comp_recommendation_results(self):
        if not hasattr(self, "comp_results_scroll"):
            return
        self.comp_results_scroll.clear()
        if hasattr(self, "comp_result_count_label"):
            self.comp_result_count_label.configure(text="")
        if not self._core_data_ready:
            ttk.Label(self.comp_results_scroll.inner, text="Core data is loading...", foreground="#666666").pack(anchor="w")
            return
        if not self.comp_recommendations:
            hint = "选择一种 Box 模式后点击“生成 Box 推荐”。"
            if self.comp_reco_mode_var.get() == "基于核心构建 Box" and self.comp_core_monsters:
                hint = "核心已更新，点击“生成 Box 推荐”以围绕这些核心生成整套 Box。"
            ttk.Label(self.comp_results_scroll.inner, text=hint, foreground="#666666", wraplength=780, justify=tk.LEFT).pack(anchor="w")
            return
        if hasattr(self, "comp_result_count_label"):
            self.comp_result_count_label.configure(text=f"{len(self.comp_recommendations)} results")
        for item in self.comp_recommendations:
            self._render_comp_recommendation_card(item)

    def _render_comp_recommendation_card(self, item):
        card = ttk.Frame(self.comp_results_scroll.inner, padding=10, style="Card.TFrame")
        card.pack(fill=tk.X, pady=(0, 10))

        top = ttk.Frame(card)
        top.pack(fill=tk.X)
        ttk.Label(top, text=item.get("name", "Box"), font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, text=f"Score {float(item.get('score', 0.0)):.4f}", foreground="#3153a1", font=("", 10, "bold")).pack(side=tk.RIGHT)

        style_text = item.get("style", "balanced")
        detail = item.get("detail", {}) or {}
        summary = f"{style_text} | size {int(detail.get('box_size', 0) or 0)} | avg support {float(detail.get('avg_support', 0.0)):.2f}"
        ttk.Label(card, text=summary, foreground="#555555").pack(anchor="w", pady=(2, 6))

        image_grid = ttk.Frame(card)
        image_grid.pack(fill=tk.X, pady=(0, 8))
        monsters = item.get("box", [])
        columns = 5
        for idx, monster in enumerate(monsters):
            box_card = self.render_image_card(
                image_grid,
                monster,
                size=(66, 66),
                subtitle=self.attr_map.get(monster, ""),
                owned=monster in set(self.my_box_state),
            )
            row = idx // columns
            col = idx % columns
            box_card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            image_grid.grid_columnconfigure(col, weight=1)

        for reason in item.get("reasons", []):
            ttk.Label(card, text=f"- {reason}", wraplength=820, justify=tk.LEFT).pack(anchor="w")

        action_row = ttk.Frame(card)
        action_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(action_row, text="复制 Box", command=lambda box=item.get("box", []): self.copy_comp_to_clipboard(box)).pack(side=tk.LEFT)
        ttk.Button(action_row, text="设前5为核心", command=lambda box=item.get("box", []): self.set_core_monsters_from_box(box)).pack(side=tk.LEFT, padx=(8, 0))

    def render_banned_section(self, parent):
        section = ttk.LabelFrame(parent, text="对方 Ban 预测", padding=8, style="Section.TLabelframe")
        section.pack(fill=tk.X, pady=(0, 10))

        if len(self.my_picks) != 5 or len(self.enemy_picks) != 5:
            if self.banned_monsters:
                preview = " / ".join(self.banned_monsters[:4])
                if len(self.banned_monsters) > 4:
                    preview += " ..."
                text = f"双方选完 5 人后显示；当前预禁：{preview}"
            else:
                text = "双方选完 5 人后显示"
            ttk.Label(section, text=text, foreground="#666666", wraplength=300, justify=tk.LEFT).pack(anchor="w")
            return

        if self.enemy_final_ban:
            text = f"对方 Ban：{self.enemy_final_ban}"
        else:
            prediction = self._predict_enemy_ban_target()
            if prediction:
                text = (
                    f"对方 Ban 预测：{prediction['monster']} "
                    f"(我方 Ban 后预估胜率 {prediction['estimated_win_rate']:.4f})"
                )
            else:
                text = "对方 Ban 预测：暂无"
        ttk.Label(section, text=text, wraplength=600, justify=tk.LEFT).pack(anchor="w")



def main():
    bootstrap_tk_env()
    root = tk.Tk()
    app = DesktopBPApp(root)
    app.set_status("桌面版已启动")
    root.mainloop()


if __name__ == "__main__":
    main()
