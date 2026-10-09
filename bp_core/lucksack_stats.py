from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from bp_core.mechanism import BPMechanismLayer, DEFAULT_MECHANISM_PATH
from bp_core.names import PROJECT_ROOT, normalize_name, slugify_name

DEFAULT_STATS_PATH = PROJECT_ROOT / "json" / "lucksack_stats.json"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "json" / "lucksack_monster_catalog.json"
DEFAULT_SWKIT_TIER_PATH = PROJECT_ROOT / "json" / "swkit_rta_tiers.json"
DEFAULT_SWRT_STATS_PATH = PROJECT_ROOT / "json" / "swrt_rta_stats.json"

RTA_TIER_BASE_SCORE = {
    "SSS": 0.62,
    "SS": 0.59,
    "S": 0.56,
    "A": 0.53,
    "B": 0.50,
    "C": 0.47,
}


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def is_usable_catalog_item(item):
    """Keep Lucksack entries that represent awakened, searchable monsters."""
    if not isinstance(item, dict) or not bool(item.get("searchable", True)):
        return False
    raw_stars = item.get("natural_stars")
    if raw_stars in (None, ""):
        return True
    stars = _as_float(raw_stars, 0.0)
    if stars >= 4:
        return True
    # Secondary-awakening entries are explicitly marked in the public
    # catalog and remain valid even though their base family is 1--3 star.
    label = " ".join(
        str(item.get(key) or "").lower()
        for key in ("label", "slabel", "slug")
    )
    return "[2a]" in label or "2a" in label.replace(" ", "")


def build_catalog_display_name_map(catalog):
    """Use full catalog labels when a short label refers to multiple units."""
    items = [
        item
        for item in (catalog if isinstance(catalog, list) else [])
        if is_usable_catalog_item(item)
    ]
    short_name_slugs = defaultdict(set)
    for item in items:
        slug = normalize_name(item.get("slug"))
        short_name = normalize_name(item.get("slabel"))
        if slug and short_name:
            short_name_slugs[short_name].add(slug)

    display_names = {}
    for item in items:
        slug = normalize_name(item.get("slug"))
        if not slug:
            continue
        short_name = str(item.get("slabel") or "").strip()
        full_name = str(item.get("label") or "").strip()
        if short_name and len(short_name_slugs[normalize_name(short_name)]) == 1:
            display_names[slug] = short_name
        else:
            display_names[slug] = full_name or short_name or slug
    return display_names


def build_catalog_ambiguous_short_names(catalog):
    display_names = build_catalog_display_name_map(catalog)
    ambiguous = set()
    for item in (catalog if isinstance(catalog, list) else []):
        if not is_usable_catalog_item(item):
            continue
        slug = normalize_name(item.get("slug"))
        short_name = normalize_name(item.get("slabel"))
        if slug and short_name and normalize_name(display_names.get(slug)) != short_name:
            ambiguous.add(short_name)
    return ambiguous


def _clamp(value, low=0.0, high=1.0):
    return max(float(low), min(float(high), float(value)))


def _combo_key(values):
    normalized = tuple(
        sorted({normalize_name(value) for value in values if normalize_name(value)})
    )
    return _combo_key_cached(normalized)


@lru_cache(maxsize=65536)
def _combo_key_cached(values):
    return "|".join(values)


def _pick_phase(turn):
    try:
        turn = int(turn)
    except (TypeError, ValueError):
        return None
    if turn <= 0 or turn > 10:
        return None
    if turn == 1:
        return 1
    if turn <= 3:
        return 2
    if turn <= 5:
        return 3
    if turn <= 7:
        return 4
    if turn <= 9:
        return 5
    return 6


class BPLucksackStats:
    """Reads the public Lucksack snapshot used as a statistical evidence layer."""

    def __init__(
        self,
        stats_path=DEFAULT_STATS_PATH,
        catalog_path=DEFAULT_CATALOG_PATH,
        rank_group=None,
        swkit_tier_path=DEFAULT_SWKIT_TIER_PATH,
        swrt_stats_path=DEFAULT_SWRT_STATS_PATH,
        mechanism_path=DEFAULT_MECHANISM_PATH,
    ):
        self.stats_path = Path(stats_path)
        self.catalog_path = Path(catalog_path)
        self.swkit_tier_path = Path(swkit_tier_path)
        self.swrt_stats_path = Path(swrt_stats_path)
        self.mechanism_path = Path(mechanism_path)
        self.data = self._load_json(self.stats_path, {})
        if not isinstance(self.data, dict):
            self.data = {}
        self.catalog = self._load_json(self.catalog_path, [])
        if not isinstance(self.catalog, list):
            self.catalog = []
        catalog_items = self.catalog if isinstance(self.catalog, list) else []
        self._usable_catalog_slugs = {
            normalize_name(item.get("slug"))
            for item in catalog_items
            if is_usable_catalog_item(item) and normalize_name(item.get("slug"))
        }
        self.swkit_tiers = self._load_json(self.swkit_tier_path, {})
        if not isinstance(self.swkit_tiers, dict):
            self.swkit_tiers = {}
        self.swrt_stats = self._load_json(self.swrt_stats_path, {})
        if not isinstance(self.swrt_stats, dict):
            self.swrt_stats = {}
        self.mechanisms = BPMechanismLayer(self.mechanism_path)
        self.rank_group = str(
            rank_group
            or os.getenv("BP_LUCKSACK_RANK_GROUP", "").strip()
            or self.data.get("default_rank_group")
            or "102"
        )
        self.display_name_map = {}
        self.image_name_map = {}
        self.name_slug_aliases = {}
        self._build_catalog_maps()
        self._solo = self._build_solo_index()
        self._duo = self._build_combo_index("duo")
        self._trio = self._build_combo_index("trio")
        self._draft_position = self._build_draft_position_index()
        self._rta_tiers = self._build_rta_tier_index()
        self._swrt_by_slug = {}
        self._swrt_id_to_slug = {}
        self._swrt_matchups = {}
        self._swrt_pair_matchups = {}
        self._build_swrt_index()
        self._summary = self._build_summary()
        self._rank_distribution = self._build_rank_distribution()
        self._resolve_cache = {}
        self._candidate_slug_cache = None
        self._score_cache = {}
        self._ban_score_cache = {}

    @staticmethod
    def _load_json(path, default):
        if not Path(path).exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return default

    @property
    def ready(self):
        statistics = self.data.get("statistics") or {}
        return bool(self.data and statistics.get("solo"))

    @property
    def source_label(self):
        if not self.ready:
            return "unavailable"
        season = self.data.get("season") or {}
        patch = self.data.get("patch") or {}
        season_name = season.get("season_name") or season.get("season_number") or "season"
        patch_name = patch.get("patch_name") or patch.get("patch_id") or "patch"
        return f"lucksack:{season_name}:{patch_name}:rank={self.rank_group}"

    def _build_catalog_maps(self):
        display_names = build_catalog_display_name_map(self.catalog)
        for item in self.catalog if isinstance(self.catalog, list) else []:
            if not is_usable_catalog_item(item):
                continue
            slug = normalize_name(item.get("slug"))
            if not slug:
                continue
            display_name = display_names.get(
                slug,
                str(item.get("slabel") or item.get("label") or slug).strip(),
            )
            label = str(item.get("label") or display_name).strip()
            self.display_name_map[slug] = display_name
            if item.get("image"):
                self.image_name_map[slug] = item["image"]
            aliases = {
                slug,
                normalize_name(display_name),
                normalize_name(label),
                slugify_name(display_name),
                slugify_name(label),
            }
            if normalize_name(item.get("slabel")) == normalize_name(display_name):
                aliases.add(normalize_name(item.get("slabel")))
            for alias in aliases:
                if alias:
                    self.name_slug_aliases.setdefault(alias, slug)

    def _rank_records(self, kind):
        statistics = self.data.get("statistics") or {}
        kind_records = statistics.get(kind) or {}
        records = kind_records.get(self.rank_group)
        if records is None:
            records = kind_records.get(str(self.data.get("default_rank_group") or "102"), [])
        return records or []

    def _build_solo_index(self):
        index = {}
        for record in self._rank_records("solo"):
            slug = normalize_name(record.get("monster_slug"))
            if slug:
                index[slug] = record
        return index

    def _build_combo_index(self, kind):
        index = {}
        for record in self._rank_records(kind):
            slugs = record.get("monster_slug")
            if not isinstance(slugs, list):
                continue
            key = _combo_key(slugs)
            if key:
                index[key] = record
        return index

    def _build_draft_position_index(self):
        dashboard = self.data.get("meta_dashboard") or {}
        groups = dashboard.get("draft_position") or {}
        phases = groups.get(self.rank_group)
        if phases is None:
            phases = groups.get(str(self.data.get("default_rank_group") or "102"), {})
        index = {}
        for phase in (phases or {}).get("phases", []):
            phase_number = _as_int(phase.get("pick_phase"), 0)
            if phase_number <= 0:
                continue
            for record in phase.get("points") or []:
                slug = normalize_name(record.get("monster_slug"))
                if slug:
                    index[(phase_number, slug)] = record
        return index

    def _build_rta_tier_index(self):
        entries = self.swkit_tiers.get("entries_by_id") or {}
        if not isinstance(entries, dict):
            entries = {}
        catalog_by_id = {
            str(item.get("com2us_id")): normalize_name(item.get("slug"))
            for item in self.catalog
            if isinstance(item, dict) and item.get("com2us_id") and item.get("slug")
        }
        return {
            catalog_by_id[unit_id]: record
            for unit_id, record in entries.items()
            if unit_id in catalog_by_id
        }

    def _build_swrt_index(self):
        if not isinstance(self.swrt_stats, dict):
            return
        self._swrt_id_to_slug = {
            str(item.get("com2us_id")): normalize_name(item.get("slug"))
            for item in self.catalog
            if isinstance(item, dict)
            and item.get("com2us_id")
            and item.get("slug")
            and is_usable_catalog_item(item)
        }
        statistics = self.swrt_stats.get("statistics") or []
        for record in statistics if isinstance(statistics, list) else []:
            if not isinstance(record, dict):
                continue
            unit_id = str(record.get("monsterId") or "")
            slug = self._swrt_id_to_slug.get(unit_id)
            if not slug:
                # When a catalog is present, an unknown ID is either an
                # unsearchable variant or stale data; do not alias it to a
                # searchable unit solely from its display name.
                if unit_id and self.catalog:
                    continue
                slug = slugify_name(
                    record.get("monsterName")
                    or record.get("monsterNameCn")
                    or record.get("monsterNameZho")
                )
                if slug and self.catalog and slug not in self._usable_catalog_slugs:
                    slug = ""
                if slug:
                    self._swrt_id_to_slug[unit_id] = slug
            if not slug:
                continue
            self._swrt_by_slug[slug] = record
            self.display_name_map.setdefault(
                slug,
                str(
                    record.get("monsterName")
                    or record.get("monsterNameCn")
                    or record.get("monsterNameZho")
                    or slug
                ).strip(),
            )
            if record.get("imageFilename"):
                self.image_name_map.setdefault(slug, record["imageFilename"])
            for alias in (
                record.get("monsterName"),
                record.get("monsterNameCn"),
                record.get("monsterNameZho"),
                unit_id,
            ):
                alias = normalize_name(alias)
                if alias:
                    self.name_slug_aliases.setdefault(alias, slug)

        highdata_by_id = self.swrt_stats.get("highdata_by_id") or {}
        for unit_id, highdata in highdata_by_id.items():
            slug = self._swrt_id_to_slug.get(str(unit_id))
            if not slug:
                continue
            for list_name in (
                "highOneVsOneList",
                "lowOneVsOneList",
                "lowOneVsOneElementalsList",
            ):
                for record in highdata.get(list_name) or []:
                    opponent_slug = self._swrt_id_to_slug.get(str(record.get("opponentMonsterId") or ""))
                    if opponent_slug:
                        self._swrt_matchups.setdefault((slug, opponent_slug), record)
            for list_name in ("highOneVsTwoList", "lowOneVsTwoList"):
                for record in highdata.get(list_name) or []:
                    opponent_slugs = [
                        self._swrt_id_to_slug.get(str(record.get("opponentMonsterOneId") or "")),
                        self._swrt_id_to_slug.get(str(record.get("opponentMonsterTwoId") or "")),
                    ]
                    if all(opponent_slugs):
                        self._swrt_pair_matchups.setdefault(
                            (slug, _combo_key(opponent_slugs)),
                            record,
                        )

    def _build_summary(self):
        dashboard = self.data.get("meta_dashboard") or {}
        summaries = dashboard.get("summary") or {}
        summary = summaries.get(self.rank_group)
        if summary is None:
            summary = summaries.get(str(self.data.get("default_rank_group") or "102"), {})
        return summary or {}

    def _build_rank_distribution(self):
        dashboard = self.data.get("meta_dashboard") or {}
        return dashboard.get("rank_distribution") or {}

    def _confidence(self, games, cap=5000):
        games = max(0.0, _as_float(games))
        if games <= 0:
            return 0.0
        return _clamp(math.log1p(games) / math.log1p(cap))

    def _solo_record(self, slug):
        return self._solo.get(normalize_name(slug))

    def _solo_score(self, slug):
        record = self._solo_record(slug)
        if not record:
            return 0.5, 0.0, None
        win_rate = _as_float(record.get("win_rate"), 0.5)
        pick_win_rate = _as_float(record.get("pick_win_rate"), win_rate)
        confidence = self._confidence(record.get("played_count"))
        score = (
            0.65 * (0.5 + (win_rate - 0.5) * confidence)
            + 0.35 * (0.5 + (pick_win_rate - 0.5) * confidence)
        )
        return _clamp(score, 0.2, 0.8), confidence, record

    def _solo_contest_score(self, slug):
        record = self._solo_record(slug)
        if not record:
            return 0.5, 0.0
        ban_rate = max(0.0, min(1.0, _as_float(record.get("ban_rate"), 0.0)))
        confidence = self._confidence(record.get("played_count"))
        return _clamp(0.5 + (ban_rate - 0.10) * confidence, 0.2, 0.8), confidence

    def score_ban_action(self, my_picks, enemy_picks, target):
        """Return the evidence fields consumed by the legacy Ban model."""
        key = (
            tuple(self.resolve_slug(value) for value in (my_picks or [])),
            tuple(self.resolve_slug(value) for value in (enemy_picks or [])),
            self.resolve_slug(target),
        )
        cached = self._ban_score_cache.get(key)
        if cached is not None:
            return cached
        target_slug = self.resolve_slug(target)
        record = self._solo_record(target_slug)
        solo_score, solo_coverage, _ = self._solo_score(target_slug)
        contest_score, contest_coverage = self._solo_contest_score(target_slug)
        matchup_score, matchup_coverage, evidence = self._swrt_matchup_score(
            [target_slug],
            [self.resolve_slug(value) for value in (my_picks or [])],
        )
        score = (
            0.55 * solo_score + 0.30 * contest_score + 0.15 * matchup_score
            if matchup_coverage > 0.0
            else 0.65 * solo_score + 0.35 * contest_score
        )
        removal = self.mechanisms.score_removal(enemy_picks, target_slug)
        result = {
            "score": 0.5,
            "mechanism_score": float(removal.get("score", 0.5)),
            "mechanism_available": bool(removal.get("available")),
            "mechanism_coverage": float(removal.get("coverage", 0.0) or 0.0),
            "mechanism_threat_reduction": float(removal.get("reduction", 0.0) or 0.0),
            "mechanism_threats": list(removal.get("threats") or []),
            "mechanism_reasons": list(removal.get("reasons") or []),
            "target_solo_score": solo_score,
            "target_solo_coverage": solo_coverage,
            "target_contest_score": contest_score,
            "target_contest_coverage": contest_coverage,
            "ban_priority_score": score,
            "target_matchup_score": matchup_score,
            "target_matchup_coverage": matchup_coverage,
            "target_solo_played": _as_int((record or {}).get("played_count")),
            "target_solo_win_rate": _as_float((record or {}).get("win_rate"), 0.5),
            "target_solo_ban_rate": _as_float((record or {}).get("ban_rate"), 0.0),
            "source": self.source_label,
            "evidence": evidence[:12],
        }
        self._ban_score_cache[key] = result
        return result

    def _combo_record(self, kind, slugs):
        key = _combo_key(slugs)
        if not key:
            return None
        return (self._duo if kind == "duo" else self._trio).get(key)

    def _combo_score(self, kind, slugs):
        record = self._combo_record(kind, slugs)
        if not record:
            return 0.5, 0.0, None
        games = record.get("played_count", record.get("appearances", 0))
        win_rate = _as_float(record.get("win_rate"), 0.5)
        if "winrate" in record:
            win_rate = _as_float(record.get("winrate"), 0.5)
        confidence = self._confidence(games, cap=3000 if kind == "duo" else 1500)
        score = 0.5 + (win_rate - 0.5) * confidence
        return _clamp(score, 0.2, 0.8), confidence, record

    def _draft_position_score(self, slug, turn):
        phase = _pick_phase(turn)
        if phase is None:
            return 0.5, 0.0, None
        record = self._draft_position.get((phase, normalize_name(slug)))
        if not record:
            return 0.5, 0.0, None
        win_rate = _as_float(record.get("win_rate"), 0.5)
        confidence = self._confidence(
            record.get("pick_count"),
            cap=50000,
        )
        score = 0.5 + (win_rate - 0.5) * confidence
        return _clamp(score, 0.2, 0.8), confidence, record

    def _rta_tier_score(self, slug):
        record = self._rta_tiers.get(normalize_name(slug))
        if not record:
            return 0.5, 0.0, None
        tier_score = RTA_TIER_BASE_SCORE.get(str(record.get("tier") or "").upper(), 0.5)
        win_rate = _as_float(record.get("win"), 50.0) / 100.0
        pick_rate = max(0.0, _as_float(record.get("pick"), 0.0))
        confidence = self._confidence(pick_rate, cap=40)
        score = (
            0.55 * tier_score
            + 0.35 * (0.5 + (win_rate - 0.5) * max(0.45, confidence))
            + 0.10 * (0.5 + 0.05 * confidence)
        )
        return _clamp(score, 0.2, 0.8), confidence, record

    def _swrt_solo_score(self, slug):
        record = self._swrt_by_slug.get(normalize_name(slug))
        if not record:
            return 0.5, 0.0, None
        win_rate = _as_float(record.get("winRate"), 0.5)
        confidence = self._confidence(record.get("pickTotal"), cap=50000)
        usage = min(1.0, max(0.0, _as_float(record.get("pickRate"), 0.0)) / 0.35)
        ban = min(1.0, max(0.0, _as_float(record.get("banRate"), 0.0)) / 0.35)
        score = (
            0.70 * (0.5 + (win_rate - 0.5) * confidence)
            + 0.20 * (0.5 + 0.04 * usage)
            + 0.10 * (0.5 + 0.03 * ban)
        )
        return _clamp(score, 0.2, 0.8), confidence, record

    def _swrt_matchup_score(self, action_slugs, enemy_slugs):
        scores = []
        coverage = []
        evidence = []
        for action_slug in action_slugs:
            for enemy_slug in enemy_slugs:
                record = self._swrt_matchups.get((action_slug, enemy_slug))
                if not record:
                    continue
                games = _as_int(record.get("pickTotal"))
                win_rate = _as_float(record.get("winRate"), 0.5)
                confidence = self._confidence(games, cap=3000)
                score = 0.5 + (win_rate - 0.5) * confidence
                scores.append(_clamp(score, 0.2, 0.8))
                coverage.append(confidence)
                evidence.append(
                    {
                        "kind": "swrt_matchup",
                        "slug": action_slug,
                        "opponent_slug": enemy_slug,
                        "played": games,
                        "win_rate": round(win_rate, 6),
                    }
                )
            if len(enemy_slugs) >= 2:
                for index in range(len(enemy_slugs)):
                    for next_index in range(index + 1, len(enemy_slugs)):
                        opponent_slugs = [enemy_slugs[index], enemy_slugs[next_index]]
                        record = self._swrt_pair_matchups.get(
                            (action_slug, _combo_key(opponent_slugs))
                        )
                        if not record:
                            continue
                        games = _as_int(record.get("pickTotal"))
                        win_rate = _as_float(record.get("winRate"), 0.5)
                        confidence = self._confidence(games, cap=2000)
                        score = 0.5 + (win_rate - 0.5) * confidence
                        scores.append(_clamp(score, 0.2, 0.8))
                        coverage.append(confidence)
                        evidence.append(
                            {
                                "kind": "swrt_team_matchup",
                                "slug": action_slug,
                                "opponent_slugs": sorted(opponent_slugs),
                                "played": games,
                                "win_rate": round(win_rate, 6),
                            }
                        )
        if not scores:
            return 0.5, 0.0, []
        return (
            _clamp(sum(scores) / len(scores), 0.2, 0.8),
            round(sum(coverage) / len(coverage), 6) if coverage else 0.0,
            evidence,
        )

    def resolve_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        cached = self._resolve_cache.get(normalized)
        if cached is not None:
            return cached
        fallback = slugify_name(value)
        slug = self.name_slug_aliases.get(normalized) or self.name_slug_aliases.get(fallback) or ""
        self._resolve_cache[normalized] = slug
        return slug

    def to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug, slug)

    def to_image_name(self, slug):
        slug = normalize_name(slug)
        return self.image_name_map.get(slug, self.to_display_name(slug))

    def candidate_slugs(self):
        """Return current RTA candidates plus searchable four-star catalog entries."""
        if self._candidate_slug_cache is None:
            slugs = set(self._solo) | set(self._swrt_by_slug)
            if self.catalog:
                slugs.intersection_update(self._usable_catalog_slugs)
            for item in self.catalog if isinstance(self.catalog, list) else []:
                if not is_usable_catalog_item(item):
                    continue
                slug = normalize_name(item.get("slug"))
                if slug:
                    slugs.add(slug)
            self._candidate_slug_cache = tuple(sorted(slugs))
        return list(self._candidate_slug_cache)

    def candidate_synergy_pairs(self, allowed, limit=32):
        """Return high-confidence duo records that should survive pair pruning."""
        allowed = {normalize_name(value) for value in (allowed or []) if value}
        if len(allowed) < 2 or not self._duo:
            return []

        candidates = []
        for key, record in self._duo.items():
            slugs = tuple(value for value in str(key).split("|") if value)
            if len(slugs) != 2 or not all(slug in allowed for slug in slugs):
                continue
            score, confidence, _record = self._combo_score("duo", slugs)
            if confidence <= 0.0:
                continue
            solo_scores = [self._solo_score(slug)[0] for slug in slugs]
            lift = score - sum(solo_scores) / len(solo_scores)
            priority = lift * confidence
            if priority <= 0.0:
                continue
            candidates.append(
                {
                    "slugs": slugs,
                    "score": round(score, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 6),
                    "priority": round(priority, 6),
                    "played": _as_int(
                        record.get("played_count", record.get("appearances"))
                    ),
                }
            )
        candidates.sort(
            key=lambda row: (row["priority"], row["confidence"], row["played"]),
            reverse=True,
        )
        return candidates[: max(0, int(limit))]

    def pick_frequency(self, slug):
        record = self._solo_record(slug)
        return _as_int((record or {}).get("played_count"))

    @staticmethod
    def _values(value):
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def score_action(self, my_picks, enemy_picks, action, turn=None):
        my_values = self._values(my_picks)
        enemy_values = self._values(enemy_picks)
        action_values = self._values(action)
        key = (
            tuple(self.resolve_slug(value) for value in my_values),
            tuple(self.resolve_slug(value) for value in enemy_values),
            tuple(self.resolve_slug(value) for value in action_values),
            None if turn is None else int(turn),
        )
        cached = self._score_cache.get(key)
        if cached is None:
            cached = self._score_action_uncached(my_values, enemy_values, action_values, turn)
            self._score_cache[key] = cached
        return cached

    def _score_action_uncached(self, my_picks, enemy_picks, action, turn=None):
        action = self._values(action)
        my_picks = self._values(my_picks)
        enemy_picks = self._values(enemy_picks)
        action_slugs = [self.resolve_slug(value) for value in action]
        action_slugs = [slug for slug in action_slugs if slug]
        own_slugs = [self.resolve_slug(value) for value in my_picks]
        all_own = list(dict.fromkeys(own_slugs + action_slugs))
        mechanism_detail = self.mechanisms.score_action(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            action=action,
        )
        if not action_slugs:
            return {
                "score": 0.5,
                "solo_score": 0.5,
                "synergy_score": 0.5,
                "coverage": 0.0,
                "swrt_solo_score": 0.5,
                "swrt_solo_coverage": 0.0,
                "matchup_score": 0.5,
                "matchup_coverage": 0.0,
                "matchup_weight": 0.0,
                "source": self.source_label,
                "evidence": [],
                "mechanism_score": float(mechanism_detail.get("score", 0.5)),
                "mechanism_available": bool(mechanism_detail.get("available")),
                "mechanism_coverage": float(mechanism_detail.get("coverage", 0.0)),
                "mechanism_coverage_gain": float(mechanism_detail.get("coverage_gain", 0.0)),
                "mechanism_reasons": list(mechanism_detail.get("reasons", [])),
                "sample_count": 0,
                "evidence_confidence": 0.0,
                "evidence_uncertainty": 1.0,
            }

        solo_scores = []
        coverage = []
        sample_counts = []
        evidence = []
        enemy_slugs = [self.resolve_slug(value) for value in (enemy_picks or [])]
        enemy_slugs = [slug for slug in enemy_slugs if slug]
        for slug in action_slugs:
            score, confidence, record = self._solo_score(slug)
            solo_scores.append(score)
            if record:
                coverage.append(confidence)
                sample_counts.append(_as_int(record.get("played_count")))
                evidence.append(
                    {
                        "kind": "solo",
                        "slug": slug,
                        "played": _as_int(record.get("played_count")),
                        "win_rate": round(_as_float(record.get("win_rate"), 0.5), 6),
                        "pick_win_rate": round(_as_float(record.get("pick_win_rate"), 0.5), 6),
                        "pick_rate": round(_as_float(record.get("played_rate"), 0.0), 6),
                        "ban_rate": round(_as_float(record.get("ban_rate"), 0.0), 6),
                        "lead_rate": round(_as_float(record.get("lead_rate"), 0.0), 6),
                    }
                )

        swrt_solo_scores = []
        swrt_solo_coverage = []
        for slug in action_slugs:
            score, confidence, record = self._swrt_solo_score(slug)
            if record:
                swrt_solo_scores.append(score)
                swrt_solo_coverage.append(confidence)
                sample_counts.append(_as_int(record.get("pickTotal")))
                evidence.append(
                    {
                        "kind": "swrt_solo",
                        "slug": slug,
                        "played": _as_int(record.get("pickTotal")),
                        "win_rate": round(_as_float(record.get("winRate"), 0.5), 6),
                        "pick_rate": round(_as_float(record.get("pickRate"), 0.0), 6),
                        "ban_rate": round(_as_float(record.get("banRate"), 0.0), 6),
                    }
                )

        synergy_scores = []
        synergy_coverage = []
        action_set = set(action_slugs)
        for duo in combinations(all_own, 2):
            if not action_set.intersection(duo):
                continue
            score, confidence, record = self._combo_score("duo", duo)
            if record:
                solo_a, _, _ = self._solo_score(duo[0])
                solo_b, _, _ = self._solo_score(duo[1])
                lift = (score - 0.5) - ((solo_a - 0.5) + (solo_b - 0.5)) / 2.0
                synergy_scores.append(0.5 + lift * confidence)
                synergy_coverage.append(confidence)
                sample_counts.append(
                    _as_int(record.get("played_count", record.get("appearances")))
                )
                evidence.append(
                    {
                        "kind": "duo",
                        "slugs": sorted(duo),
                        "played": _as_int(record.get("played_count", record.get("appearances"))),
                        "win_rate": round(_as_float(record.get("win_rate", record.get("winrate")), 0.5), 6),
                    }
                )

        rta_scores = []
        rta_coverage = []
        for slug in action_slugs:
            score, confidence, record = self._rta_tier_score(slug)
            if record:
                rta_scores.append(score)
                rta_coverage.append(confidence)
                evidence.append(
                    {
                        "kind": "swkit_rta_tier",
                        "slug": slug,
                        "tier": record.get("tier"),
                        "pick_rate": round(_as_float(record.get("pick"), 0.0), 6),
                        "win_rate": round(_as_float(record.get("win"), 50.0) / 100.0, 6),
                        "tier_date": self.swkit_tiers.get("tier_date", ""),
                    }
                )

        if len(all_own) >= 3:
            for trio in combinations(all_own, 3):
                if not action_set.intersection(trio):
                    continue
                score, confidence, record = self._combo_score("trio", trio)
                if record:
                    synergy_scores.append(0.5 + (score - 0.5) * confidence)
                    synergy_coverage.append(confidence)
                    sample_counts.append(
                        _as_int(record.get("played_count", record.get("appearances")))
                    )
                    evidence.append(
                        {
                            "kind": "trio",
                            "slugs": sorted(trio),
                            "played": _as_int(record.get("played_count", record.get("appearances"))),
                            "win_rate": round(_as_float(record.get("win_rate", record.get("winrate")), 0.5), 6),
                        }
                    )

        phase_scores = []
        phase_coverage = []
        if turn is not None:
            for slug in action_slugs:
                score, confidence, record = self._draft_position_score(slug, turn)
                if record:
                    phase_scores.append(score)
                    phase_coverage.append(confidence)
                    sample_counts.append(_as_int(record.get("pick_count")))
                    evidence.append(
                        {
                            "kind": "draft_position",
                            "phase": _pick_phase(turn),
                            "slug": slug,
                            "played": _as_int(record.get("pick_count")),
                            "win_rate": round(_as_float(record.get("win_rate"), 0.5), 6),
                            "pick_rate": round(_as_float(record.get("pick_rate"), 0.0), 6),
                        }
                    )

        solo_score = sum(solo_scores) / len(solo_scores)
        synergy_score = (
            sum(synergy_scores) / len(synergy_scores)
            if synergy_scores
            else 0.5
        )
        phase_score = (
            sum(phase_scores) / len(phase_scores)
            if phase_scores
            else 0.5
        )
        rta_tier_score = (
            sum(rta_scores) / len(rta_scores)
            if rta_scores
            else 0.5
        )
        swrt_solo_score = (
            sum(swrt_solo_scores) / len(swrt_solo_scores)
            if swrt_solo_scores
            else 0.5
        )
        if swrt_solo_scores:
            solo_score = _clamp(0.78 * solo_score + 0.22 * swrt_solo_score, 0.2, 0.8)
        matchup_score, matchup_coverage, matchup_evidence = self._swrt_matchup_score(
            action_slugs,
            enemy_slugs,
        )
        evidence.extend(matchup_evidence)
        for threat in mechanism_detail.get("threats", [])[:4]:
            if threat.get("gain", 0.0) <= 0:
                continue
            evidence.append(
                {
                    "kind": "mechanism_counter",
                    "threat": threat.get("threat"),
                    "label": threat.get("label"),
                    "gain": round(_as_float(threat.get("gain")), 6),
                    "responses": list(threat.get("responses") or []),
                    "enemy_sources": list(threat.get("enemy_sources") or []),
                }
            )
        base_score = _clamp(
            (
                0.55 * solo_score
                + 0.22 * synergy_score
                + 0.13 * phase_score
                + 0.10 * rta_tier_score
            )
            if phase_scores and rta_scores
            else (
                0.60 * solo_score
                + 0.25 * synergy_score
                + 0.15 * phase_score
                if phase_scores
                else (
                    0.63 * solo_score
                    + 0.27 * synergy_score
                    + 0.10 * rta_tier_score
                    if rta_scores
                    else 0.70 * solo_score + 0.30 * synergy_score
                )
            ),
            0.2,
            0.8,
        )
        matchup_weight = 0.0
        if matchup_coverage > 0:
            matchup_weight = 0.10 + 0.10 * min(1.0, matchup_coverage)
        score = _clamp(
            (1.0 - matchup_weight) * base_score + matchup_weight * matchup_score,
            0.2,
            0.8,
        )
        data_coverage = round(
            0.50
            * max(
                sum(coverage) / len(coverage) if coverage else 0.0,
                sum(synergy_coverage) / len(synergy_coverage)
                if synergy_coverage
                else 0.0,
                sum(swrt_solo_coverage) / len(swrt_solo_coverage)
                if swrt_solo_coverage
                else 0.0,
                matchup_coverage,
            )
            + 0.25
            * (
                sum(phase_coverage) / len(phase_coverage)
                if phase_coverage
                else 0.0
            )
            + 0.25
            * (
                sum(rta_coverage) / len(rta_coverage)
                if rta_coverage
                else 0.0
            ),
            6,
        )
        return {
            "score": round(score, 6),
            "solo_score": round(solo_score, 6),
            "synergy_score": round(synergy_score, 6),
            "draft_position_score": round(phase_score, 6),
            "coverage": round(sum(coverage) / len(coverage), 6) if coverage else 0.0,
            "draft_position_coverage": (
                round(sum(phase_coverage) / len(phase_coverage), 6)
                if phase_coverage
                else 0.0
            ),
            "rta_tier_score": round(rta_tier_score, 6),
            "rta_tier_coverage": (
                round(sum(rta_coverage) / len(rta_coverage), 6)
                if rta_coverage
                else 0.0
            ),
            "swrt_solo_score": round(swrt_solo_score, 6),
            "swrt_solo_coverage": (
                round(sum(swrt_solo_coverage) / len(swrt_solo_coverage), 6)
                if swrt_solo_coverage
                else 0.0
            ),
            "matchup_score": round(matchup_score, 6),
            "matchup_coverage": round(matchup_coverage, 6),
            "matchup_weight": round(matchup_weight, 6),
            "mechanism_score": float(mechanism_detail.get("score", 0.5)),
            "mechanism_available": bool(mechanism_detail.get("available")),
            "mechanism_coverage": float(mechanism_detail.get("coverage", 0.0)),
            "mechanism_coverage_before": float(mechanism_detail.get("coverage_before", 0.0)),
            "mechanism_coverage_gain": float(mechanism_detail.get("coverage_gain", 0.0)),
            "mechanism_threats": mechanism_detail.get("threats", []),
            "mechanism_action_tags": mechanism_detail.get("action_tags", []),
            "mechanism_reasons": list(mechanism_detail.get("reasons", [])),
            "summary_first_pick_win_rate": _as_float(
                self._summary.get("first_pick_win_rate"),
                0.5,
            ),
            "summary_second_pick_win_rate": _as_float(
                self._summary.get("second_pick_win_rate"),
                0.5,
            ),
            "summary_total_battles": _as_int(self._summary.get("total_battles")),
            "rank_distribution": self._rank_distribution,
            "rank_group": self.rank_group,
            "season": self.data.get("season") or {},
            "patch": self.data.get("patch") or {},
            "data_coverage": data_coverage,
            "sample_count": max(sample_counts, default=0),
            "evidence_confidence": data_coverage,
            "evidence_uncertainty": round(1.0 - data_coverage, 6),
            "source": self.source_label,
            "evidence": evidence[:12],
        }
