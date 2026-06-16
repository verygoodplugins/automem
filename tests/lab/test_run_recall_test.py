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
    assert card["config_name"] == "cfg"
    assert card["ndcg_10"] == 0.5
    assert card["distractor_rate_10"] == 0.25
    assert card["latency_ms"] == 150.0
    assert card["complexity"] == 7
