from bp_recommender_from_samples import (
    load_samples,
    BPRecommenderFromSamples,
    pretty_print,
    infer_turn_from_state
)

def main():
    samples = load_samples("bp_training_samples_me_only.json")
    recommender = BPRecommenderFromSamples(samples)

    # 你改这里
    my_picks = ["Ereshion"]
    enemy_picks = ["Oliver", "Shi Hou"]
    first_pick_side = "me"

    turn = infer_turn_from_state(my_picks, enemy_picks, first_pick_side)

    print("=== 当前状态 ===")
    print("my_picks        =", my_picks)
    print("enemy_picks     =", enemy_picks)
    print("first_pick_side =", first_pick_side)
    print("inferred_turn   =", turn)
    print()

    if turn is None:
        print("当前不是我方选人回合，无法推荐。")
        return

    recs = recommender.recommend(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        first_pick_side=first_pick_side,
        turn=turn,
        top_k=5,
        min_similarity=3.0,
        min_support=1
    )

    print("=== 推荐结果 ===")
    pretty_print(recs)

if __name__ == "__main__":
    main()