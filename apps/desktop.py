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
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageGrab, ImageOps, ImageTk
from sw_ui_detection.capture import save_screen_capture

ROOT_DIR = Path(__file__).resolve().parent.parent
MY_BOX_PATH = ROOT_DIR / "json" / "my_box.json"
IMAGE_MAP_PATH = ROOT_DIR / "json" / "monster_images.json"
ATTR_MAP_PATH = ROOT_DIR / "json" / "monster_attributes.json"
FAMILY_MAP_PATH = ROOT_DIR / "json" / "monster_families.json"
RELATION_META_V2_PATH = ROOT_DIR / "models" / "bp_relation_meta_v2.json"
RELATION_VOCAB_V2_PATH = ROOT_DIR / "models" / "bp_relation_vocab_v2.json"
RELATION_MODEL_V2_PATH = ROOT_DIR / "models" / "bp_relation_net_v2.pt"
RELATION_META_LEGACY_PATH = ROOT_DIR / "models" / "bp_relation_meta.json"
RELATION_VOCAB_LEGACY_PATH = ROOT_DIR / "models" / "bp_relation_vocab.json"
RELATION_MODEL_LEGACY_PATH = ROOT_DIR / "models" / "bp_relation_net.pt"
RELATION_V2_BUNDLE_READY = all(
    path.exists()
    for path in (RELATION_MODEL_V2_PATH, RELATION_META_V2_PATH, RELATION_VOCAB_V2_PATH)
)
RELATION_META_PATH = (
    RELATION_META_V2_PATH if RELATION_V2_BUNDLE_READY else RELATION_META_LEGACY_PATH
)
RELATION_VOCAB_PATH = (
    RELATION_VOCAB_V2_PATH if RELATION_V2_BUNDLE_READY else RELATION_VOCAB_LEGACY_PATH
)
SW_UI_DIR = ROOT_DIR / "sw_ui_detection"
SW_UI_OUTPUT_DIR = SW_UI_DIR / "output"
APP_BP_IMPORT_PATH = SW_UI_OUTPUT_DIR / "app_bp_import_latest.png"
APP_BP_JSON_PATH = SW_UI_OUTPUT_DIR / "app_bp_capture_latest.json"

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

    from bp_core.lucksack_stats import BPLucksackStats

    luck_stats = BPLucksackStats()
    for slug in luck_stats.candidate_slugs():
        display_name = luck_stats.to_display_name(slug)
        if not display_name or display_name in seen:
            continue
        seen.add(display_name)
        monster_pool.append(display_name)
        pick_freq[display_name] = luck_stats.pick_frequency(slug)

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
    first_pick_side = normalize_name(first_pick_side)
    if first_pick_side not in {"me", "enemy"}:
        return None

    if first_pick_side == "me":
        my_turns = {1, 4, 5, 8, 9}
    else:
        my_turns = {2, 3, 6, 7, 10}

    if turn not in my_turns:
        return None
    return turn


def infer_side_for_order(order, first_pick_side):
    if order is None or order < 1 or order > 10:
        return None
    first_pick_side = normalize_name(first_pick_side)
    if first_pick_side not in {"me", "enemy"}:
        return None
    first = "my" if first_pick_side == "me" else "enemy"
    other = "enemy" if first == "my" else "my"
    return [first, other, other, first, first, other, other, first, first, other][order - 1]


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

        self.relation_recommender = None
        self.ban_value_recommender = None
        self.prefix_value_recommender = None
        self.bp_analyzer = None
        self._core_data_ready = False
        self._box_tab_loaded = False
        self._lazy_bootstrap_started = False

        self.monster_pool = []
        self.attr_map = {}
        self.family_map = {}
        self.image_lookup = {}
        self.pick_freq = Counter()

        self.photo_cache = {}
        self.my_box_state = sorted(set(load_json_file(MY_BOX_PATH, [])))
        self.my_picks = []
        self.enemy_picks = []
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
        self.bp_recognition_result = None
        self.bp_recognition_source = None
        self.bp_recognition_image_path = None
        self._screen_capture_hidden = False
        self.bp_enemy_candidate_top_n_var = tk.IntVar(value=4)

        self.status_var = tk.StringVar(value="就绪")
        self.first_pick_var = tk.StringVar(value="me")
        self.search_var = tk.StringVar(value="")
        self.only_owned_in_picker_var = tk.BooleanVar(value=False)
        self.use_my_box_for_reco_var = tk.BooleanVar(value=True)
        self.only_with_image_var = tk.BooleanVar(value=False)
        self.attr_filter_var = tk.StringVar(value="全部")
        self.family_filter_var = tk.StringVar(value="全部")
        self.top_k_var = tk.IntVar(value=6)
        self.bp_page_size_var = tk.IntVar(value=BP_GALLERY_COLUMNS * BP_GALLERY_PAGE_ROWS)
        self.box_page_size_var = tk.IntVar(value=50)
        self.my_leader_var = tk.StringVar(value="")
        self.enemy_leader_var = tk.StringVar(value="")
        self.box_search_var = tk.StringVar(value="")
        self.box_attr_filter_var = tk.StringVar(value="全部")
        self.box_only_owned_var = tk.BooleanVar(value=False)
        self.box_only_with_image_var = tk.BooleanVar(value=True)

        self.bp_gallery_page = 1
        self.box_gallery_page = 1
        self.attributes = ["全部"]
        self.families = ["全部"]

        self._configure_style()
        self._build_ui()
        self._bind_var_events()
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
        from bp_core.lucksack_stats import BPLucksackStats

        luck_stats = BPLucksackStats()
        for slug in luck_stats.candidate_slugs():
            display_name = luck_stats.to_display_name(slug)
            if not display_name:
                continue
            attr_map.setdefault(display_name, "")
        image_lookup = build_image_lookup(image_map)
        monster_pool, pick_freq = load_neural_monster_metadata()
        if not monster_pool:
            fallback_names = set(image_map) | set(attr_map) | set(family_map)
            monster_pool = sorted(str(name).strip() for name in fallback_names if str(name).strip())
        return {
            "monster_pool": monster_pool,
            "pick_freq": pick_freq,
            "attr_map": attr_map,
            "family_map": family_map,
            "image_lookup": image_lookup,
        }

    def _apply_lazy_bootstrap_result(self, result):
        if isinstance(result, dict) and result.get("error") is not None:
            print(result.get("traceback", ""))
            self.set_status(f"Core data load failed: {result['error']}")
            return

        self.monster_pool = result["monster_pool"]
        self.pick_freq = result["pick_freq"]
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
        for widget_name in ("bp_attr_filter_combo", "box_attr_filter_combo"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(values=self.attributes)
        for widget_name in ("bp_family_filter_combo",):
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
        if selected == str(self.box_tab):
            return "box"
        return "bp"

    def _on_tab_changed(self, _event=None):
        tab_name = self._current_tab_name()
        if tab_name == "box" and not self._box_tab_loaded:
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
        tk.Label(info, text=f"Win {self._format_plan_win_rate(plan)}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)
        rerank_value = plan.get("rerank_value", plan.get("detail", {}).get("partial_value_win_rate"))
        if rerank_value is None:
            rerank_text = "N/A"
        else:
            try:
                rerank_text = f"{float(rerank_value):.4f}"
            except (TypeError, ValueError):
                rerank_text = "N/A"
        tk.Label(info, text=f"Rerank {rerank_text}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)

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
        tk.Label(info, text=f"Win {self._format_plan_win_rate(plan)}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)
        tk.Label(info, text=f"Gain {float(detail.get('win_rate_gain', 0.0)):+.4f}", bg="#ffffff", anchor="w", font=("", 8)).pack(fill=tk.X)

        tk.Button(card, text="Apply Ban", command=lambda m=monster: self.apply_my_final_ban(m), font=("", 8, "bold"), height=1).pack(fill=tk.X, padx=6, pady=(0, 6))
        return card

    def _format_plan_win_rate(self, plan):
        detail = plan.get("detail", {}) if isinstance(plan.get("detail"), dict) else {}
        source = str(detail.get("win_rate_source") or "")
        if source in {"prefix_value_unavailable", "prefix_value_failed"}:
            return "N/A"
        if source == "" and str(plan.get("source", "")).startswith("relation"):
            return "N/A"
        try:
            return f"{float(plan.get('estimated_win_rate', 0.0)):.4f}"
        except (TypeError, ValueError):
            return "N/A"

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
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")

        current_box_frame = ttk.LabelFrame(left, text="当前 Box 已有魔灵", padding=8, style="Section.TLabelframe")
        current_box_frame.pack(fill=tk.BOTH, pady=(0, 8))
        self.current_box_scroll = ScrollableFrame(current_box_frame)
        self.current_box_scroll.pack(fill=tk.BOTH, expand=True)
        self.current_box_scroll.canvas.configure(height=310)

        self.box_gallery_scroll = ScrollableFrame(left)
        self.box_gallery_scroll.pack(fill=tk.BOTH, expand=True)

    def on_bp_filter_changed(self):
        self.bp_gallery_page = 1
        self.refresh_bp_gallery()

    def on_box_filter_changed(self):
        self.box_gallery_page = 1
        if self._box_tab_loaded or self._current_tab_name() == "box":
            self.refresh_box_tab()

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

    def _set_transient_status(self, text):
        self.status_var.set(text)

    def _relation_model_bundle_ready(self):
        return (
            (
                (ROOT_DIR / "models" / "bp_relation_net_v2.pt").exists()
                and (ROOT_DIR / "models" / "bp_relation_meta_v2.json").exists()
                and (ROOT_DIR / "models" / "bp_relation_vocab_v2.json").exists()
            )
            or (
                (ROOT_DIR / "models" / "bp_relation_net.pt").exists()
                and (ROOT_DIR / "models" / "bp_relation_meta.json").exists()
                and (ROOT_DIR / "models" / "bp_relation_vocab.json").exists()
            )
        )

    def _get_relation_recommender(self):
        if self.relation_recommender is not None:
            return self.relation_recommender
        if not self._relation_model_bundle_ready():
            raise RuntimeError("Relation model is not available. Run python -m training.relation first.")
        from bp_recommender.relation import BPRelationRecommender

        self.relation_recommender = BPRelationRecommender()
        return self.relation_recommender

    def _prefix_value_model_bundle_ready(self):
        return (
            (ROOT_DIR / "models" / "bp_prefix_value_net.pt").exists()
            and (ROOT_DIR / "models" / "bp_prefix_value_meta.json").exists()
            and (ROOT_DIR / "models" / "bp_prefix_value_vocab.json").exists()
        )

    def _get_prefix_value_recommender(self):
        if self.prefix_value_recommender is not None:
            return self.prefix_value_recommender
        if not self._prefix_value_model_bundle_ready():
            raise RuntimeError("Prefix value model is not available. Run python -m training.prefix_value first.")
        from bp_recommender.prefix_value import BPPrefixValueRecommender

        self.prefix_value_recommender = BPPrefixValueRecommender()
        return self.prefix_value_recommender

    def _get_ban_value_recommender(self):
        if self.ban_value_recommender is not None:
            return self.ban_value_recommender
        from bp_recommender.ban_value import BPBanValueRecommender

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
        first_pick_side = normalize_name(self.first_pick_var.get())
        if first_pick_side not in {"me", "enemy"}:
            return None
        first = "my" if first_pick_side == "me" else "enemy"
        other = "enemy" if first == "my" else "my"
        pattern = [first, other, other, first, first, other, other, first, first, other]
        return pattern[order - 1]

    def _current_pick_side(self):
        return self._pick_side_for_order(self._current_pick_order())

    def _effective_bp_page_size(self):
        raw_size = max(BP_GALLERY_COLUMNS, int(self.bp_page_size_var.get()))
        return int(math.ceil(raw_size / BP_GALLERY_COLUMNS) * BP_GALLERY_COLUMNS)

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

    def _box_gallery_primary_buttons(self, monster, owned):
        if owned:
            return [("Remove", lambda m=monster: self.toggle_my_box(m))]
        return [("Add", lambda m=monster: self.toggle_my_box(m))]

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

        self._restore_after_screen_capture()
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
        self.bp_recognition_result = result
        self.bp_recognition_source = result.get("source")
        self.bp_recognition_image_path = Path(result["image_path"]) if result.get("image_path") else None
        self.set_status("BP recognition finished")
        self.refresh_bp_state_panel()

    def _restore_after_screen_capture(self):
        if not self._screen_capture_hidden:
            return
        self._screen_capture_hidden = False
        self.root.deiconify()
        self.root.lift()

    def recognize_bp_from_screen(self):
        if self._active_task_name:
            self._set_transient_status(f"{self._active_task_name} in progress...")
            return False

        candidate_top_k = max(1, int(self.bp_enemy_candidate_top_n_var.get()))
        self._screen_capture_hidden = True
        self.root.withdraw()

        def work():
            time.sleep(0.15)
            save_screen_capture(APP_BP_IMPORT_PATH)
            return self._run_bp_recognition_from_path(
                APP_BP_IMPORT_PATH,
                "system screen capture",
                candidate_top_k,
            )

        started = self._start_background_task(
            "Recognizing current screen",
            work,
            self._apply_bp_recognition_result,
        )
        if not started:
            self._restore_after_screen_capture()
        return started

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
        self.refresh_bp_state_panel()
        self.set_status("Cleared BP recognition result")

    def refresh_all(self):
        self.refresh_sidebar_pick_plans()
        self.refresh_bp_gallery()
        self.refresh_bp_state_panel()
        current_tab = self._current_tab_name()
        if self._box_tab_loaded or current_tab == "box":
            self.refresh_box_tab()

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
        ttk.Button(toolbar_row, text="Capture Current Screen", command=self.recognize_bp_from_screen).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
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
        occupied = set(self.my_picks) | set(self.enemy_picks)
        return [m for m in self.monster_pool if m not in occupied]




    def build_sidebar_prediction_card(self, parent, prediction, idx):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.RIDGE)
        title_label = tk.Label(card, text=f"Pred {idx}", bg="#ffffff", font=("", 9, "bold"))
        title_label.pack(anchor="w", padx=6, pady=(5, 3))

        monsters = [monster for monster in prediction.get("plan") or prediction.get("monsters") or [] if monster]
        if not monsters and prediction.get("monster"):
            monsters = [prediction.get("monster")]
        image_row = tk.Frame(card, bg="#ffffff")
        image_row.pack(fill=tk.X, padx=6)
        clickable_widgets = [card, title_label, image_row]
        for monster in monsters:
            block = tk.Frame(image_row, bg="#ffffff")
            block.pack(side=tk.LEFT, padx=2, pady=(0, 3))
            photo = self.get_photo(monster, size=SIDEBAR_PLAN_IMAGE_SIZE)
            image_label = tk.Label(block, image=photo if photo else "", text="" if photo else "No Img", bg="#ffffff")
            image_label.image = photo
            image_label.pack()
            short_name = monster if len(monster) <= 6 else f"{monster[:5]}..."
            name_label = tk.Label(
                block,
                text=short_name,
                bg="#ffffff",
                font=("", 7),
                wraplength=48,
                justify=tk.CENTER,
            )
            name_label.pack()
            clickable_widgets.extend([block, image_label, name_label])

        info = tk.Frame(card, bg="#ffffff")
        info.pack(fill=tk.X, padx=6, pady=(4, 4))
        info_label = tk.Label(info, text=f"Score {float(prediction.get('final_score', 0.0)):.4f}", bg="#ffffff", anchor="w", font=("", 8))
        info_label.pack(fill=tk.X)
        clickable_widgets.extend([info, info_label])
        tk.Button(
            card,
            text="Apply",
            command=lambda p=prediction: self.apply_enemy_prediction(p),
            font=("", 8, "bold"),
            height=1,
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        for widget in clickable_widgets:
            widget.bind("<Button-1>", lambda _event, p=prediction: self.apply_enemy_prediction(p))
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
        target = self.my_picks if side == "my" else self.enemy_picks
        if monster in target:
            return
        if len(target) >= 5:
            messagebox.showwarning("提示", f"{'我方' if side == 'my' else '敌方'}已满，无法继续加入。")
            return

        self.remove_from_all(monster)
        target.append(monster)
        final_status = status_text or f"已将 {monster} 加入 {'我方' if side == 'my' else '敌方'}"
        self._apply_bp_state_change(final_status)

    def remove_one(self, monster):
        self.remove_from_all(monster)
        self._apply_bp_state_change(f"已移除 {monster}")

    def clear_all(self):
        self.my_picks = []
        self.enemy_picks = []
        self.my_final_ban = None
        self.enemy_final_ban = None
        self.my_leader_var.set("")
        self.enemy_leader_var.set("")
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

    def apply_enemy_prediction(self, prediction):
        targets = [x for x in prediction.get("plan") or prediction.get("monsters") or [] if x]
        if not targets and prediction.get("monster"):
            targets = [prediction.get("monster")]
        remain = 5 - len(self.enemy_picks)
        unique_targets = []
        for monster in targets:
            if monster not in unique_targets:
                unique_targets.append(monster)

        to_add = [monster for monster in unique_targets if monster not in self.enemy_picks]
        if len(to_add) > remain:
            messagebox.showwarning("提示", f"敌方剩余位置不足，仅剩 {remain} 个。")
            return

        for monster in to_add:
            self.remove_from_all(monster)
            self.enemy_picks.append(monster)

        joined = " + ".join(to_add) if to_add else "当前预测"
        self._apply_bp_state_change(f"已采用对方预测：{joined}")

    def apply_my_final_ban(self, monster):
        self.my_final_ban = monster
        self._apply_bp_state_change(f"已采用我方 Ban：{monster}")

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
                top_k=(
                    state_snapshot["top_k"] * 4
                    if self._prefix_value_model_bundle_ready()
                    else state_snapshot["top_k"]
                ),
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
            prefix_value_recommender = self._get_prefix_value_recommender()
            base_eval = prefix_value_recommender.evaluate_state(
                my_picks=state_snapshot["my_picks"],
                enemy_picks=state_snapshot["enemy_picks"],
                turn=state_snapshot["turn"],
                first_pick_side=state_snapshot["first_pick_side"],
                current_side="me",
                my_leader=state_snapshot["my_leader"],
                enemy_leader=state_snapshot["enemy_leader"],
                return_detail=True,
            )
            base_wr = float(base_eval.get("calibrated_win_rate", 0.0))
            base_raw_wr = float(base_eval.get("raw_win_rate", 0.0))
            value_source = "prefix_value"
        except Exception as exc:
            print(f"[WARN] prefix value model unavailable for pick ranking: {exc}")
            for plan in plans:
                detail = dict(plan.get("detail", {}) or {})
                detail["win_rate_source"] = "prefix_value_unavailable"
                detail["ranking_source"] = "relation_only"
                plan["detail"] = detail
            return plans[: int(state_snapshot["top_k"])]

        enriched = []
        for plan in plans:
            item = dict(plan)
            plan_picks = [monster for monster in item.get("plan", []) if monster]
            future_my_picks = list(state_snapshot["my_picks"]) + plan_picks
            detail = dict(item.get("detail", {}) or {})
            try:
                eval_result = prefix_value_recommender.evaluate_state(
                    my_picks=future_my_picks,
                    enemy_picks=state_snapshot["enemy_picks"],
                    turn=state_snapshot["turn"] + len(plan_picks),
                    first_pick_side=state_snapshot["first_pick_side"],
                    current_side="enemy",
                    my_leader=state_snapshot["my_leader"],
                    enemy_leader=state_snapshot["enemy_leader"],
                    return_detail=True,
                )
                wr = float(eval_result.get("calibrated_win_rate", 0.0))
                raw_wr = float(eval_result.get("raw_win_rate", 0.0))
                relation_component = float(
                    detail.get("normalized_relation_score", item.get("plan_score", 0.0)) or 0.0
                )
                draft_stats_component = float(detail.get("draft_stats_score", 0.5) or 0.5)
                fused_score = (
                    0.65 * wr + 0.25 * relation_component + 0.10 * draft_stats_component
                    if detail.get("draft_stats_source") not in {None, "unavailable"}
                    else 0.70 * wr + 0.30 * relation_component
                )
                item["plan_score"] = round(fused_score, 6)
                item["estimated_win_rate"] = round(wr, 6)
                item["rerank_value"] = round(wr, 6)
                detail.update(
                    {
                        "pre_pick_win_rate": round(base_wr, 6),
                        "post_pick_win_rate": round(wr, 6),
                        "win_rate_gain": round(wr - base_wr, 6),
                        "raw_pre_pick_win_rate": round(base_raw_wr, 6),
                        "raw_post_pick_win_rate": round(raw_wr, 6),
                        "raw_win_rate_gain": round(raw_wr - base_raw_wr, 6),
                        "win_rate_source": value_source,
                        "ranking_source": (
                            "prefix_value_0.65_relation_0.25_stats_0.10"
                            if detail.get("draft_stats_source") not in {None, "unavailable"}
                            else "prefix_value_0.70_relation_0.30"
                        ),
                        "relation_component": round(relation_component, 6),
                        "value_component": round(wr, 6),
                        "draft_stats_component": round(draft_stats_component, 6),
                        "partial_value_win_rate": round(wr, 6),
                        "fused_score": round(fused_score, 6),
                    }
                )
            except Exception as exc:
                print(f"[WARN] failed to evaluate pick plan win rate for {plan_picks}: {exc}")
                item["estimated_win_rate"] = 0.0
                detail.setdefault("win_rate_source", "prefix_value_failed")
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row.get("plan_score", 0.0)), reverse=True)
        return enriched[: int(state_snapshot["top_k"])]

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
            "turn": len(self.my_picks) + len(self.enemy_picks) + 1,
            "first_pick_side": self.first_pick_var.get(),
            "top_k": max(1, int(self.top_k_var.get())),
            "candidate_pool": list(self.build_candidate_pool()),
            "my_leader": self.my_leader_var.get().strip(),
            "enemy_leader": self.enemy_leader_var.get().strip(),
            "context_signature": self._recommendation_context_signature(),
        }

        def work():
            relation_recommender = self._get_relation_recommender()
            predictions = relation_recommender.recommend_enemy_next_picks(
                my_picks=state_snapshot["my_picks"],
                enemy_picks=state_snapshot["enemy_picks"],
                first_pick_side=state_snapshot["first_pick_side"],
                top_k=(
                    state_snapshot["top_k"] * 4
                    if self._prefix_value_model_bundle_ready()
                    else state_snapshot["top_k"]
                ),
                candidate_pool=state_snapshot["candidate_pool"],
                max_pair_candidates=400,
                my_leader=state_snapshot["my_leader"],
                enemy_leader=state_snapshot["enemy_leader"],
            )
            return self._attach_enemy_prediction_values(predictions, state_snapshot)

        def apply_result(predictions):
            if state_snapshot["context_signature"] != self._recommendation_context_signature():
                self.set_status("Enemy predictions were discarded because the BP state changed")
                return
            self.enemy_predictions = predictions
            self.set_status(f"Generated {len(predictions)} enemy predictions")
            self.refresh_sidebar_pick_plans()
            self.refresh_bp_state_panel()

        return self._start_background_task("Generating enemy predictions", work, apply_result)

    def _attach_enemy_prediction_values(self, predictions, state_snapshot):
        if not predictions:
            return predictions
        try:
            prefix_value_recommender = self._get_prefix_value_recommender()
        except Exception as exc:
            print(f"[WARN] prefix value model unavailable for enemy ranking: {exc}")
            return predictions[: int(state_snapshot["top_k"])]

        enriched = []
        for prediction in predictions:
            item = dict(prediction)
            monsters = [monster for monster in item.get("plan") or item.get("monsters") or [] if monster]
            if not monsters:
                monster = item.get("monster") or item.get("display_name")
                monsters = [monster] if monster else []
            future_enemy_picks = list(state_snapshot["enemy_picks"]) + monsters
            detail = dict(item.get("detail", {}) or {})
            try:
                eval_result = prefix_value_recommender.evaluate_state(
                    my_picks=state_snapshot["my_picks"],
                    enemy_picks=future_enemy_picks,
                    turn=state_snapshot["turn"] + len(monsters),
                    first_pick_side=state_snapshot["first_pick_side"],
                    current_side=(
                        "me"
                        if infer_side_for_order(
                            state_snapshot["turn"] + len(monsters),
                            state_snapshot["first_pick_side"],
                        )
                        == "my"
                        else "enemy"
                    ),
                    my_leader=state_snapshot["my_leader"],
                    enemy_leader=state_snapshot["enemy_leader"],
                    return_detail=True,
                )
                wr = float(eval_result.get("calibrated_win_rate", 0.0))
                relation_component = float(
                    detail.get("normalized_relation_score", item.get("final_score", 0.0)) or 0.0
                )
                draft_stats_component = float(detail.get("draft_stats_score", 0.5) or 0.5)
                adversarial_score = (
                    0.65 * (1.0 - wr) + 0.25 * relation_component + 0.10 * draft_stats_component
                    if detail.get("draft_stats_source") not in {None, "unavailable"}
                    else 0.70 * (1.0 - wr) + 0.30 * relation_component
                )
                item["estimated_win_rate"] = round(wr, 6)
                item["final_score"] = round(adversarial_score, 6)
                detail.update(
                    {
                        "post_enemy_pick_win_rate": round(wr, 6),
                        "enemy_adversarial_score": round(adversarial_score, 6),
                        "relation_component": round(relation_component, 6),
                        "win_rate_source": "prefix_value",
                        "ranking_source": (
                            "enemy_value_0.65_relation_0.25_stats_0.10"
                            if detail.get("draft_stats_source") not in {None, "unavailable"}
                            else "enemy_value_0.70_relation_0.30"
                        ),
                        "draft_stats_component": round(draft_stats_component, 6),
                    }
                )
            except Exception as exc:
                print(f"[WARN] failed to evaluate enemy prediction {monsters}: {exc}")
                detail.setdefault("win_rate_source", "prefix_value_failed")
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
        return enriched[: int(state_snapshot["top_k"])]

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
        self.box_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bp_tab, text="BP 模式")
        self.notebook.add(self.box_tab, text="我的 Box 管理")

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

    def render_banned_section(self, parent):
        section = ttk.LabelFrame(parent, text="对方 Ban 预测", padding=8, style="Section.TLabelframe")
        section.pack(fill=tk.X, pady=(0, 10))

        if len(self.my_picks) != 5 or len(self.enemy_picks) != 5:
            ttk.Label(
                section,
                text="双方选完 5 人后显示",
                foreground="#666666",
                wraplength=300,
                justify=tk.LEFT,
            ).pack(anchor="w")
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
