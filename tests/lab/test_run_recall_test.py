import run_recall_test as rr


def test_build_scorecard_reads_the_four_axes():
    result = rr.TestRunResult(config_name="cfg")
    result.complexity = 7
    result.query_results = [
        rr.QueryResult(
            query="q1",
            expected_ids=["a"],
            retrieved_ids=["a", "d1"],
            ndcg_10=1.0,
            distractor_rate_10=0.5,
            latency_ms=100.0,
        ),
        rr.QueryResult(
            query="q2",
            expected_ids=["b"],
            retrieved_ids=["b"],
            ndcg_10=0.0,
            distractor_rate_10=0.0,
            latency_ms=200.0,
        ),
    ]
    card = rr.build_scorecard(result)
    assert card["name"] == "cfg"
    assert card["ndcg_10"] == 0.5
    assert card["distractor_rate_10"] == 0.25
    assert card["latency_ms"] == 150.0
    assert card["complexity"] == 7


def test_build_scorecard_output_feeds_pick_winner():
    """The producer/consumer key contract: build_scorecard cards must be
    directly consumable by pick_winner (the path Plan B's matrix harness uses)."""
    import lab_metrics as m

    result = rr.TestRunResult(config_name="baseline")
    result.complexity = 5
    result.query_results = [
        rr.QueryResult(
            query="q",
            expected_ids=["a"],
            retrieved_ids=["a"],
            ndcg_10=0.8,
            distractor_rate_10=0.1,
            latency_ms=100.0,
        ),
    ]
    card = rr.build_scorecard(result)
    winner = m.pick_winner([card], baseline_name="baseline")
    assert winner["name"] == "baseline"
