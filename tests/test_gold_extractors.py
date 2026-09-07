from kgu.data.nba import Example
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.graph import BiTemporalGraph
from kgu.ops import diff
from kgu.types import Op, OpKind
from tests.conftest import PLAYS


def _ex(before, after, mentioned):
    return Example(0, "trade", "2021", "messi joins inter", mentioned, before, after)


def test_gold_all_returns_full_diff(before, after):
    ex = _ex(before, after, ["Messi", "Inter"])
    assert GoldAllExtractor().extract(ex, BiTemporalGraph.from_triples(before)) == diff(before, after)


def test_gold_explicit_keeps_only_ops_between_mentioned_entities(before, after):
    ex = _ex(before, after, ["Messi", "Inter", "Barca"])
    ops = GoldExplicitExtractor().extract(ex, BiTemporalGraph.from_triples(before))
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
        Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
    }
