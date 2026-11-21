"""Regression tests: keyword extraction must keep non-ASCII letters."""

from __future__ import annotations

import pytest

from automem.utils.text import _extract_keywords


@pytest.mark.parametrize(
    "query, expected",
    [
        ("café menu", ["café", "menu"]),
        ("résumé review", ["résumé", "review"]),
        ("Überprüfung der Rechnung", ["überprüfung", "der", "rechnung"]),
        ("naïve bayes classifier", ["naïve", "bayes", "classifier"]),
        ("项目 部署 计划", ["项目", "部署", "计划"]),
    ],
)
def test_non_ascii_words_stay_whole(query, expected):
    assert _extract_keywords(query) == expected


def test_ascii_behavior_unchanged():
    assert _extract_keywords("The deploy-plan for api_v2 is ok") == ["deploy-plan", "api_v2"]
    assert _extract_keywords("") == []
