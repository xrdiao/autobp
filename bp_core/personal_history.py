import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import combinations

from bp_core.data_common import NameResolver
from bp_core.transformer_common import normalize_name, slugify_fallback


class _WeightedStat:
    def __init__(self):
        self.wins = 0.0
        self.games = 0.0

    def add(self, won, weight=1.0):
        weight = max(0.0, float(weight))
        self.games += weight
        if won:
            self.wins += weight

    def win_rate(self, prior_win_rate=0.5, prior_count=8.0):
        return (self.wins + prior_win_rate * prior_count) / max(1e-9, self.games + prior_count)


class BPPersonalHistoryModel:
    def __init__(
        self,
        prior_win_rate=0.5,
        prior_count=8.0,
        half_life_days=45.0,
        confidence_sample_count=30.0,
        min_personal_samples=5.0,
        exposure_cap=50.0,
        high_exposure_threshold=40.0,
        small_lift_threshold=0.03,
        deduplicate=True,
        alias_map=None,
        display_name_map=None,
    ):
        self.prior_win_rate = float(prior_win_rate)
        self.prior_count = float(prior_count)
        self.half_life_days = float(half_life_days)
        self.confidence_sample_count = float(confidence_sample_count)
        self.min_personal_samples = float(min_personal_samples)
        self.exposure_cap = float(exposure_cap)
        self.high_exposure_threshold = float(high_exposure_threshold)
        self.small_lift_threshold = float(small_lift_threshold)
        self.deduplicate = bool(deduplicate)
        self.resolver = NameResolver(alias_map=alias_map, display_name_map=display_name_map)
        self.clear()

    def clear(self):
        self.seen_keys = set()
        self.records = []
        self.overall = _WeightedStat()
        self.my_pick_stats = {}
        self.my_pair_stats = {}
        self.matchup_stats = {}
        self.enemy_stats = {}
        self.my_ban_stats = {}
        self.enemy_present_not_banned_stats = {}
        self.win_presence = {}
        self.loss_presence = {}
        self.total_win_weight = 0.0
        self.total_loss_weight = 0.0

    def update_records(self, records, deduplicate=None):
        deduplicate = self.deduplicate if deduplicate is None else bool(deduplicate)
        for record in records or []:
            if not isinstance(record, dict):
                continue
            key = self._record_key(record)
            if deduplicate and key in self.seen_keys:
                continue

            my_picks = self._slug_list(record.get("my_picks"))
            enemy_picks = self._slug_list(record.get("enemy_picks"))
            result = normalize_name(record.get("result"))
            if not my_picks or not enemy_picks or result not in {"win", "won", "victory", "loss", "lose", "lost", "defeat"}:
                continue

            won = result in {"win", "won", "victory"}
            weight = self._time_weight(record.get("timestamp"))
            my_ban = self._to_slug(record.get("my_ban"))
            self.seen_keys.add(key)
            self.records.append(record)
            self.overall.add(won, weight)

            unique_my = sorted(set(my_picks))
            unique_enemy = sorted(set(enemy_picks))
            for slug in unique_my:
                self._stat(self.my_pick_stats, slug).add(won, weight)
                if won:
                    self.win_presence[slug] = self.win_presence.get(slug, 0.0) + weight
                    self.total_win_weight += weight
                else:
                    self.loss_presence[slug] = self.loss_presence.get(slug, 0.0) + weight
                    self.total_loss_weight += weight

            for first, second in combinations(unique_my, 2):
                self._stat(self.my_pair_stats, self._pair_key(first, second)).add(won, weight)

            for my_slug in unique_my:
                for enemy_slug in unique_enemy:
                    self._stat(self.matchup_stats, (my_slug, enemy_slug)).add(won, weight)

            for enemy_slug in unique_enemy:
                self._stat(self.enemy_stats, enemy_slug).add(won, weight)
                if my_ban and my_ban == enemy_slug:
                    self._stat(self.my_ban_stats, enemy_slug).add(won, weight)
                else:
                    self._stat(self.enemy_present_not_banned_stats, enemy_slug).add(won, weight)

    def score_pick_action(self, my_picks, enemy_picks, action):
        my_slugs = self._slug_list(my_picks)
        enemy_slugs = self._slug_list(enemy_picks)
        action_slugs = self._slug_list(action)
        signals = []

        for slug in action_slugs:
            stat = self.my_pick_stats.get(slug)
            if stat and stat.games > 0:
                wr = stat.win_rate(self.prior_win_rate, self.prior_count)
                signals.append(self._signal_from_win_rate(wr, stat.games, f"{self._label(slug)} personal pick"))

            win_presence = self.win_presence.get(slug, 0.0)
            loss_presence = self.loss_presence.get(slug, 0.0)
            if win_presence + loss_presence > 0:
                win_share = win_presence / max(1e-9, self.total_win_weight)
                loss_share = loss_presence / max(1e-9, self.total_loss_weight)
                presence_lift = max(-0.25, min(0.25, 2.0 * (win_share - loss_share)))
                signals.append(
                    {
                        "score": 0.5 + presence_lift,
                        "lift": presence_lift,
                        "sample_count": win_presence + loss_presence,
                        "reason": f"{self._label(slug)} win/loss lineup presence lift {presence_lift:+.3f}",
                        "kind": "positive" if presence_lift > 0 else "negative",
                    }
                )

        all_my = sorted(set(my_slugs + action_slugs))
        for first, second in combinations(all_my, 2):
            if first not in action_slugs and second not in action_slugs:
                continue
            stat = self.my_pair_stats.get(self._pair_key(first, second))
            if stat and stat.games > 0:
                wr = stat.win_rate(self.prior_win_rate, self.prior_count)
                signals.append(self._signal_from_win_rate(wr, stat.games, f"{self._label(first)} + {self._label(second)} synergy"))

        for action_slug in action_slugs:
            matchup_values = []
            matchup_games = 0.0
            for enemy_slug in enemy_slugs:
                stat = self.matchup_stats.get((action_slug, enemy_slug))
                if stat and stat.games > 0:
                    matchup_values.append(stat.win_rate(self.prior_win_rate, self.prior_count) * stat.games)
                    matchup_games += stat.games
            if matchup_games > 0:
                wr = sum(matchup_values) / matchup_games
                signals.append(self._signal_from_win_rate(wr, matchup_games, f"{self._label(action_slug)} vs current enemy history"))

        return self._finalize(signals)

    def score_ban_action(self, my_picks, enemy_picks, ban_candidate):
        candidate = self._to_slug(ban_candidate)
        if not candidate:
            return self._finalize([])
        signals = []

        facing_stat = self.enemy_stats.get(candidate)
        if facing_stat and facing_stat.games > 0:
            facing_wr = facing_stat.win_rate(self.prior_win_rate, self.prior_count)
            ban_lift = self._global_win_rate() - facing_wr
            signals.append(self._signal_from_lift(ban_lift, facing_stat.games, f"Facing {self._label(candidate)}"))

        banned_stat = self.my_ban_stats.get(candidate)
        if banned_stat and banned_stat.games > 0:
            wr = banned_stat.win_rate(self.prior_win_rate, self.prior_count)
            signals.append(self._signal_from_win_rate(wr, banned_stat.games, f"Banning {self._label(candidate)}"))

        not_banned_stat = self.enemy_present_not_banned_stats.get(candidate)
        if not_banned_stat and not_banned_stat.games > 0:
            wr = not_banned_stat.win_rate(self.prior_win_rate, self.prior_count)
            ban_lift = self._global_win_rate() - wr
            signals.append(self._signal_from_lift(ban_lift, not_banned_stat.games, f"Not banning {self._label(candidate)}"))

        return self._finalize(signals)

    def _finalize(self, signals):
        if not signals:
            return {
                "personal_score": 0.5,
                "confidence": 0.0,
                "sample_count": 0.0,
                "reasons": ["insufficient personal evidence: no personal history, using relation score"],
            }

        raw_sample_count = sum(max(0.0, float(signal.get("sample_count", 0.0))) for signal in signals)
        effective_sample_count = min(raw_sample_count, self.exposure_cap)
        if raw_sample_count < self.min_personal_samples:
            return {
                "personal_score": 0.5,
                "confidence": min(0.10, effective_sample_count / max(1e-9, self.confidence_sample_count)),
                "sample_count": float(effective_sample_count),
                "raw_sample_count": float(raw_sample_count),
                "reasons": [
                    "insufficient personal evidence: personal history is limited; keeping neutral adjustment",
                ],
            }

        weighted_lift = 0.0
        if raw_sample_count > 0:
            weighted_lift = sum(
                float(signal.get("lift", 0.0)) * max(0.0, float(signal.get("sample_count", 0.0)))
                for signal in signals
            ) / raw_sample_count
        if raw_sample_count > self.high_exposure_threshold and abs(weighted_lift) < self.small_lift_threshold:
            reasons = [
                "overexposure neutralized: common personal action but no clear win-rate lift; neutralized",
            ]
            reasons.extend(self._rank_reasons(signals)[:2])
            return {
                "personal_score": 0.5,
                "confidence": min(1.0, effective_sample_count / max(1e-9, self.confidence_sample_count)),
                "sample_count": float(effective_sample_count),
                "raw_sample_count": float(raw_sample_count),
                "reasons": reasons,
            }

        if raw_sample_count <= 0:
            score = 0.5
        else:
            score = sum(
                float(signal.get("score", 0.5)) * max(0.0, float(signal.get("sample_count", 0.0)))
                for signal in signals
            ) / raw_sample_count
        confidence = min(1.0, effective_sample_count / max(1e-9, self.confidence_sample_count))
        reasons = self._rank_reasons(signals)
        return {
            "personal_score": float(max(0.0, min(1.0, score))),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "sample_count": float(effective_sample_count),
            "raw_sample_count": float(raw_sample_count),
            "reasons": reasons,
        }

    def _global_win_rate(self):
        return self.overall.win_rate(self.prior_win_rate, self.prior_count)

    def _signal_from_win_rate(self, win_rate, sample_count, label):
        return self._signal_from_lift(float(win_rate) - self._global_win_rate(), sample_count, label)

    def _signal_from_lift(self, lift, sample_count, label):
        lift = max(-0.25, min(0.25, float(lift)))
        kind = "positive" if lift > 0 else "negative"
        if abs(lift) < 1e-9:
            kind = "neutral"
        return {
            "score": 0.5 + lift,
            "lift": lift,
            "sample_count": float(sample_count),
            "reason": f"{kind} personal evidence: {label} lift {lift:+.3f} over {float(sample_count):.1f} exposure",
            "kind": kind,
        }

    def _rank_reasons(self, signals):
        reasons = []
        for signal in sorted(signals, key=lambda item: abs(float(item.get("lift", 0.0))) * float(item.get("sample_count", 0.0)), reverse=True):
            reason = signal.get("reason")
            if reason:
                reasons.append(reason)
            if len(reasons) >= 5:
                break
        return reasons

    def _stat(self, store, key):
        if key not in store:
            store[key] = _WeightedStat()
        return store[key]

    def _to_slug(self, value):
        if value is None:
            return ""
        return self.resolver.to_slug(value) or slugify_fallback(value)

    def _slug_list(self, values):
        if not values:
            return []
        return [slug for slug in (self._to_slug(value) for value in values) if slug]

    def _pair_key(self, first, second):
        return tuple(sorted((first, second)))

    def _label(self, slug):
        return str(slug).replace("-", " ").title()

    def _record_key(self, record):
        match_id = record.get("match_id")
        if match_id:
            return f"match:{match_id}"
        payload = {
            "timestamp": record.get("timestamp"),
            "my_picks": record.get("my_picks"),
            "enemy_picks": record.get("enemy_picks"),
            "my_ban": record.get("my_ban"),
            "enemy_ban": record.get("enemy_ban"),
            "result": record.get("result"),
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return "hash:" + hashlib.sha1(data.encode("utf-8")).hexdigest()

    def _time_weight(self, timestamp):
        dt = self._parse_timestamp(timestamp)
        if dt is None:
            return 1.0
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
        return math.exp(-age_days / max(1e-9, self.half_life_days))

    def _parse_timestamp(self, timestamp):
        if not timestamp:
            return None
        if isinstance(timestamp, datetime):
            return timestamp
        text = str(timestamp).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        for parser in (
            lambda value: datetime.fromisoformat(value),
            lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M:%S"),
            lambda value: datetime.strptime(value, "%Y-%m-%d"),
        ):
            try:
                return parser(text)
            except ValueError:
                continue
        return None
