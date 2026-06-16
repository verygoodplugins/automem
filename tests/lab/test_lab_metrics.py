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


def test_config_complexity_counts_active_knobs():
    baseline = {
        "SEARCH_WEIGHT_VECTOR": "0.35",
        "SEARCH_WEIGHT_KEYWORD": "0.35",
        "SEARCH_WEIGHT_RELEVANCE": "0.0",  # off -> not counted
    }
    assert m.config_complexity(baseline) == 2

    simpler = {
        "SEARCH_WEIGHT_VECTOR": "0.35",
        "SEARCH_WEIGHT_KEYWORD": "0.0",
        "SEARCH_WEIGHT_RELEVANCE": "0.0",
    }
    assert m.config_complexity(simpler) == 1

    flags_and_gates = {
        "SEARCH_WEIGHT_VECTOR": "0.35",  # +1
        "ENRICHMENT_ENABLED": "true",  # +1
        "JIT_ENRICHMENT_ENABLED": "false",  # +0
        "RECALL_RECENCY_BIAS": "off",  # +0
        "RECALL_RELEVANCE_GATE": "0.2",  # +1 (gate > 0)
        "SEARCH_TAG_SCORE_TOKEN_CAP": "0",  # +0
    }
    assert m.config_complexity(flags_and_gates) == 3


def _card(name, ndcg, distractor, latency, complexity):
    return {
        "name": name,
        "ndcg_10": ndcg,
        "distractor_rate_10": distractor,
        "latency_ms": latency,
        "complexity": complexity,
    }


def test_pick_winner_prefers_simpler_within_ndcg_tolerance():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("complex", 0.803, 0.10, 120, 13),  # tiny ndcg gain, more knobs
        _card("simple", 0.801, 0.10, 90, 8),  # within tol, fewer knobs + faster
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "simple"


def test_pick_winner_rejects_precision_regression():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("greedy", 0.900, 0.30, 100, 11),  # big ndcg but junk floods results
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "baseline"


def test_pick_winner_picks_clear_quality_jump():
    cards = [
        _card("baseline", 0.800, 0.10, 100, 11),
        _card("better", 0.860, 0.09, 100, 11),  # real gain, no regression
    ]
    winner = m.pick_winner(cards, baseline_name="baseline")
    assert winner["name"] == "better"
