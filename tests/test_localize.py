from kgu.localize import AddCandidate, CutCandidate, localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import COACH, FRIEND, PLAYS, SPONSOR, TEAM

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def _run(graph):
    apply_ops(graph, EXPLICIT, at=1)
    return localize(graph, EXPLICIT, at=1)


def test_cut_side_finds_common_neighbour_edges(graph):
    loc = _run(graph)
    # Pedri kề cả Messi và Barca → mọi cạnh Pedri–{Messi,Barca} là ứng viên
    assert loc.cut_triples == {
        ("Pedri", PLAYS, "Barca"),
        ("Pedri", TEAM, "Messi"),
        ("Messi", TEAM, "Pedri"),
        ("Pedri", FRIEND, "Messi"),
    }
    pedri = [c for c in loc.cut if c.hub == "Pedri"]
    assert all(c.anchors == frozenset({"Messi", "Barca"}) for c in pedri)


def test_cut_side_ignores_one_leg_entities(graph):
    loc = _run(graph)
    assert ("Adidas", SPONSOR, "Messi") not in loc.cut_triples   # chỉ nối 1 participant
    assert ("Xavi", COACH, "Barca") not in loc.cut_triples       # chỉ nối Barca


def test_add_side_lists_new_context_members(graph):
    loc = _run(graph)
    assert loc.add_pairs == {
        frozenset({"Messi", "Lautaro"}),
        frozenset({"Messi", "Inzaghi"}),
    }
    lautaro = next(c for c in loc.add if c.neighbor == "Lautaro")
    assert lautaro == AddCandidate("Messi", "Lautaro", ("Lautaro", PLAYS, "Inter"))


def test_add_side_skips_neighbours_already_linked(graph):
    graph.add(("Messi", FRIEND, "Lautaro"), valid_from=0)
    loc = _run(graph)
    assert frozenset({"Messi", "Lautaro"}) not in loc.add_pairs


def test_no_ops_no_candidates(graph):
    loc = localize(graph, [], at=1)
    assert loc.cut == [] and loc.add == []


def test_results_are_deterministic(graph, before):
    from kgu.graph import BiTemporalGraph
    a = _run(graph)
    b = _run(BiTemporalGraph.from_triples(before))
    assert a == b
