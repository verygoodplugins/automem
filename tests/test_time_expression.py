"""Regression tests for out-of-range ``last N days`` time expressions."""

from __future__ import annotations

from datetime import datetime

import pytest

from automem.utils.time import _parse_time_expression


def test_last_n_days_valid_range():
    start, end = _parse_time_expression("last 7 days")
    assert start is not None and end is not None
    assert (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days == 7


@pytest.mark.parametrize(
    "expression",
    ["last -3 days", "last 1000000 days", "last 99999999999 days"],
)
def test_last_n_days_out_of_range_is_ignored(expression):
    assert _parse_time_expression(expression) == (None, None)
