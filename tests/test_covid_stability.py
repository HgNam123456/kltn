import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("dateparser")

spec = importlib.util.spec_from_file_location("covid_stability", Path(__file__).parents[1] / "scripts" / "covid_stability.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_key_normalises_case_underscores_and_date_formats():
    a = ("WHO", "declares_emergency", "Global Emergency", ["2020-01-30"], [])
    b = ("who", "declares emergency", "global  emergency", ["30-01-2020"], None)
    assert st.key(a) == st.key(b)
    assert st.key(a, with_time=False) == ("who", "declares emergency", "global emergency")


def test_jaccard_skips_groups_where_both_runs_are_empty():
    assert st.jaccard(set(), set()) is None
    assert st.jaccard({1, 2}, set()) == 0.0
    assert st.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert st.stats([0.5, None, 1.0]) == {"mean": 0.75, "std": 0.25, "groups": 2}
