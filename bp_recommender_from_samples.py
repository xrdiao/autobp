import json
from collections import defaultdict
from itertools import combinations

def load_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

def normalize_name(name: str) -> str:
    return (name or "").strip()

def is_my_double_pick_turn(turn, first_pick_side):
    if first_pick_side == "me":
        return turn in {4, 8}
    return turn in {2, 6}

class BPRecommenderFromSamples:
    def __init__(self, samples):
        self.samples = samples
        self.monster_pool = self._collect_monster_pool()

    def _collect_monster_pool(self):
        pool = set()
        for sample in self.samples:
            for x in sample.get("my_picks", []):
                pool.add(normalize_name(x))
            for x in sample.get("enemy_picks", []):
                pool.add(normalize_name(x))
            if sample.get("next_pick"):
                pool.add(normalize_name(sample["next_pick"]))
            if sample.get("my_ban"):
                pool.add(normalize_name(sample["my_ban"]))
            if sample.get("enemy_ban"):
                pool.add(normalize_name(sample["enemy_ban"]))
        return sorted(x for x in pool if x)

    def _overlap_score(self, a, b):
        return len(set(a) & set(b))

    def _state_similarity(self, query, sample):
        score = 0.0

        if query["first_pick_side"] == sample.get("first_pick_side"):
            score += 2.0

        turn_diff = abs(query["turn"] - sample.get("turn", 0))
        score += max(0, 3.0 - turn_diff)

        my_overlap = self._overlap_score(query["my_picks"], sample.get("my_picks", []))
        score += my_overlap * 3.0

        enemy_overlap = self._overlap_score(query["enemy_picks"], sample.get("enemy_picks", []))
        score += enemy_overlap * 3.0

        if set(query["my_picks"]) == set(sample.get("my_picks", [])):
            score += 2.0
        if set(query["enemy_picks"]) == set(sample.get("enemy_picks", [])):
            score += 2.0

        return score

    def _final_score_from_stats(self, weighted_count, weighted_win):
        estimated_win_rate = weighted_win / weighted_count if weighted_count > 0 else 0.0
        final_score = 0.7 * weighted_count + 0.3 * (estimated_win_rate * 10)
        return final_score, estimated_win_rate

    def recommend_next_picks(
        self,
        my_picks,
        enemy_picks,
        first_pick_side,
        turn=None,
        top_k=10,
        min_similarity=3.0,
        min_support=1,
        banned_monsters=None,
        owned_monsters=None
    ):
        if turn is None:
            turn = infer_turn_from_state(my_picks, enemy_picks, first_pick_side)

        if turn is None:
            return []

        my_picks = [normalize_name(x) for x in my_picks]
        enemy_picks = [normalize_name(x) for x in enemy_picks]
        banned_monsters = set(normalize_name(x) for x in (banned_monsters or []))
        owned_monsters = set(normalize_name(x) for x in (owned_monsters or [])) if owned_monsters else None

        query = {
            "my_picks": my_picks,
            "enemy_picks": enemy_picks,
            "first_pick_side": first_pick_side,
            "turn": turn
        }

        already_unavailable = set(my_picks) | set(enemy_picks) | banned_monsters

        stats = defaultdict(lambda: {
            "weighted_count": 0.0,
            "weighted_win": 0.0,
            "matched_samples": 0,
            "sample_ids": [],
            "match_ids": set(),
        })

        for sample in self.samples:
            candidate = normalize_name(sample.get("next_pick"))
            if not candidate:
                continue

            if candidate in already_unavailable:
                continue

            if owned_monsters is not None and candidate not in owned_monsters:
                continue

            sim = self._state_similarity(query, sample)
            if sim < min_similarity:
                continue

            stats[candidate]["weighted_count"] += sim
            if sample.get("result") == "win":
                stats[candidate]["weighted_win"] += sim
            stats[candidate]["matched_samples"] += 1
            stats[candidate]["sample_ids"].append(sample.get("sample_id"))
            stats[candidate]["match_ids"].add(sample.get("match_id"))

        recommendations = []
        for monster, s in stats.items():
            if s["matched_samples"] < min_support:
                continue

            final_score, estimated_win_rate = self._final_score_from_stats(
                s["weighted_count"], s["weighted_win"]
            )

            recommendations.append({
                "monster": monster,
                "final_score": round(final_score, 4),
                "weighted_count": round(s["weighted_count"], 4),
                "estimated_win_rate": round(estimated_win_rate, 4),
                "matched_samples": s["matched_samples"],
                "matched_matches": len(s["match_ids"]),
                "sample_ids": s["sample_ids"][:10]
            })

        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        return recommendations[:top_k]

    def recommend_pick_plans(
        self,
        my_picks,
        enemy_picks,
        first_pick_side,
        turn=None,
        top_k=5,
        min_similarity=3.0,
        min_support=1,
        banned_monsters=None,
        owned_monsters=None,
        candidate_pool_size=12
    ):
        """
        输出多套 pick 方案：
        - 单选回合：每个 monster 是一套方案
        - 双选回合：从 top 单卡里组合成双 pick 方案
        """
        if turn is None:
            turn = infer_turn_from_state(my_picks, enemy_picks, first_pick_side)

        if turn is None:
            return []

        single_recs = self.recommend_next_picks(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            first_pick_side=first_pick_side,
            turn=turn,
            top_k=max(candidate_pool_size, top_k),
            min_similarity=min_similarity,
            min_support=min_support,
            banned_monsters=banned_monsters,
            owned_monsters=owned_monsters
        )

        if not is_my_double_pick_turn(turn, first_pick_side):
            plans = []
            for rec in single_recs[:top_k]:
                plans.append({
                    "plan": [rec["monster"]],
                    "plan_score": rec["final_score"],
                    "estimated_win_rate": rec["estimated_win_rate"],
                    "matched_samples": rec["matched_samples"],
                    "matched_matches": rec["matched_matches"],
                    "detail": [rec]
                })
            return plans

        # 双选方案：两两组合
        plans = []
        for a, b in combinations(single_recs[:candidate_pool_size], 2):
            if a["monster"] == b["monster"]:
                continue

            # 简单组合评分，可后续升级
            combo_score = a["final_score"] + b["final_score"]
            combo_wr = (a["estimated_win_rate"] + b["estimated_win_rate"]) / 2.0
            combo_support = a["matched_samples"] + b["matched_samples"]
            combo_matches = a["matched_matches"] + b["matched_matches"]

            plans.append({
                "plan": [a["monster"], b["monster"]],
                "plan_score": round(combo_score, 4),
                "estimated_win_rate": round(combo_wr, 4),
                "matched_samples": combo_support,
                "matched_matches": combo_matches,
                "detail": [a, b]
            })

        plans.sort(key=lambda x: x["plan_score"], reverse=True)
        return plans[:top_k]

    def _ban_similarity(self, my_picks, enemy_picks, my_ban_candidate, sample):
        """
        用完整阵容 + ban 来估计 ban 后结果
        sample 为完整对局展开样本中的任意一条，只要它携带整场信息即可
        """
        score = 0.0

        score += self._overlap_score(my_picks, sample.get("my_picks", [])) * 3.0
        score += self._overlap_score(enemy_picks, sample.get("enemy_picks", [])) * 3.0

        if set(my_picks) == set(sample.get("my_picks", [])):
            score += 4.0
        if set(enemy_picks) == set(sample.get("enemy_picks", [])):
            score += 4.0

        if normalize_name(sample.get("enemy_ban")) == normalize_name(my_ban_candidate):
            score += 5.0

        return score

    def recommend_bans(
        self,
        my_picks,
        enemy_picks,
        top_k=5,
        min_similarity=5.0,
        min_support=1
    ):
        """
        在选人结束后，评估 ban 对方哪个单位更优。
        约定：
        - my_picks: 我方5只
        - enemy_picks: 对方5只
        返回推荐 ban 对象（通常从 enemy_picks 中挑）
        """
        my_picks = [normalize_name(x) for x in my_picks]
        enemy_picks = [normalize_name(x) for x in enemy_picks]

        if len(my_picks) != 5 or len(enemy_picks) != 5:
            return []

        results = []

        for ban_target in enemy_picks:
            weighted_count = 0.0
            weighted_win = 0.0
            matched_samples = 0
            match_ids = set()
            sample_ids = []

            for sample in self.samples:
                sim = self._ban_similarity(my_picks, enemy_picks, ban_target, sample)
                if sim < min_similarity:
                    continue

                weighted_count += sim
                if sample.get("result") == "win":
                    weighted_win += sim
                matched_samples += 1
                match_ids.add(sample.get("match_id"))
                sample_ids.append(sample.get("sample_id"))

            if matched_samples < min_support:
                continue

            final_score, estimated_win_rate = self._final_score_from_stats(weighted_count, weighted_win)

            results.append({
                "ban_target": ban_target,
                "final_score": round(final_score, 4),
                "estimated_win_rate": round(estimated_win_rate, 4),
                "weighted_count": round(weighted_count, 4),
                "matched_samples": matched_samples,
                "matched_matches": len(match_ids),
                "sample_ids": sample_ids[:10]
            })

        results.sort(key=lambda x: x["estimated_win_rate"], reverse=True)
        return results[:top_k]
    
    def recommend_enemy_next_picks(
        self,
        my_picks,
        enemy_picks,
        first_pick_side,
        top_k=5,
        min_similarity=3.0,
        min_support=1,
        banned_monsters=None,
        owned_monsters=None
    ):
        """
        通过“视角翻转”近似预测对方下一手常见 pick。
        原理：
        - 把当前 enemy 当成样本中的 my
        - 把当前 my 当成样本中的 enemy
        - 翻转 first_pick_side
        - 然后复用 recommend_next_picks
        """
        flipped_first_pick_side = "enemy" if first_pick_side == "me" else "me"

        return self.recommend_next_picks(
            my_picks=enemy_picks,
            enemy_picks=my_picks,
            first_pick_side=flipped_first_pick_side,
            turn=None,
            top_k=top_k,
            min_similarity=min_similarity,
            min_support=min_support,
            banned_monsters=banned_monsters,
            owned_monsters=owned_monsters
        )