from kgu.graph import BiTemporalGraph

T_MB = ("Messi", "plays_for", "Barca")
T_MI = ("Messi", "plays_for", "Inter")
T_PB = ("Pedri", "plays_for", "Barca")


def test_from_triples_all_active_at_0():
    g = BiTemporalGraph.from_triples([T_MB, T_PB])
    assert g.active(0) == {T_MB, T_PB}
    assert g.entities(0) == {"Messi", "Barca", "Pedri"}


def test_invalidate_keeps_history_and_hides_after():
    g = BiTemporalGraph.from_triples([T_MB])
    assert g.invalidate(T_MB, valid_to=1) is True
    assert g.is_active(T_MB, 0) is True      # lịch sử vẫn truy vấn được
    assert g.is_active(T_MB, 1) is False
    assert g.active(1) == set()
    assert g.invalidate(T_MB, valid_to=1) is False  # đã inactive


def test_add_then_readd_is_noop_when_active():
    g = BiTemporalGraph()
    assert g.add(T_MI, valid_from=1) is True
    assert g.add(T_MI, valid_from=1) is False
    assert g.active(0) == set()
    assert g.active(1) == {T_MI}


def test_invalidate_then_readd_creates_second_interval():
    g = BiTemporalGraph.from_triples([T_MB])
    g.invalidate(T_MB, 1)
    g.add(T_MB, 2)
    hist = g.history(T_MB)
    assert [(e.valid_from, e.valid_to) for e in hist] == [(0, 1), (2, None)]
    assert g.is_active(T_MB, 1) is False and g.is_active(T_MB, 2) is True


def test_neighbors_and_edges_of_respect_time():
    g = BiTemporalGraph.from_triples([T_MB, T_PB])
    g.invalidate(T_MB, 1)
    g.add(T_MI, 1)
    assert g.neighbors("Messi", 0) == {"Barca"}
    assert g.neighbors("Messi", 1) == {"Inter"}
    assert g.edges_of("Barca", 1) == {T_PB}
    assert g.neighbors("Barca", 1) == {"Pedri"}


def test_to_records():
    g = BiTemporalGraph.from_triples([T_MB])
    g.invalidate(T_MB, 1)
    assert g.to_records() == [
        {"h": "Messi", "r": "plays_for", "t": "Barca", "valid_from": 0, "valid_to": 1}
    ]
