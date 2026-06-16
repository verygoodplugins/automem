import lab_metrics as m


def test_recall_at_k_counts_hits_in_top_k():
    assert m.recall_at_k(["a", "b", "c"], ["c"], 5) == 1.0
    assert m.recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert m.recall_at_k([], ["c"], 5) == 0.0


def test_mrr_uses_first_hit_rank():
    assert m.mrr(["a", "b", "c"], ["b"]) == 0.5
    assert m.mrr(["a", "b", "c"], ["z"]) == 0.0


def test_ndcg_at_k_rewards_top_rank():
    top = m.ndcg_at_k(["x", "a", "b"], ["x"], 10)
    buried = m.ndcg_at_k(["a", "b", "x"], ["x"], 10)
    assert top == 1.0
    assert 0.0 < buried < top


def test_distractor_rate_counts_distractors_in_top_k():
    retrieved = ["good", "d1", "d2", "good2"]
    distractors = {"d1", "d2"}
    assert m.distractor_rate_at_k(retrieved, distractors, 4) == 0.5
    # top-1 is clean -> 0.0
    assert m.distractor_rate_at_k(retrieved, distractors, 1) == 0.0
    # all distractors -> 1.0
    assert m.distractor_rate_at_k(["d1", "d2"], distractors, 10) == 1.0
    # empties are safe
    assert m.distractor_rate_at_k([], distractors, 10) == 0.0
    assert m.distractor_rate_at_k(retrieved, distractors, 0) == 0.0
