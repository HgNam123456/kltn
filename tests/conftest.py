import pytest

from kgu.graph import BiTemporalGraph

# Thế giới nhỏ: Messi rời Barca sang Inter.
PLAYS = "plays_for"
TEAM = "teammate"
COACH = "head_coach"
FRIEND = "close_friend"
SPONSOR = "sponsor"

BEFORE = {
    ("Messi", PLAYS, "Barca"),
    ("Pedri", PLAYS, "Barca"),
    ("Pedri", TEAM, "Messi"),
    ("Messi", TEAM, "Pedri"),
    ("Xavi", COACH, "Barca"),
    ("Pedri", FRIEND, "Messi"),
    ("Adidas", SPONSOR, "Messi"),        # một chân: chỉ nối Messi
    ("Lautaro", PLAYS, "Inter"),
    ("Inzaghi", COACH, "Inter"),
}
AFTER = (BEFORE - {
    ("Messi", PLAYS, "Barca"),
    ("Pedri", TEAM, "Messi"),
    ("Messi", TEAM, "Pedri"),
}) | {
    ("Messi", PLAYS, "Inter"),
    ("Lautaro", TEAM, "Messi"),
    ("Messi", TEAM, "Lautaro"),
}


@pytest.fixture
def before():
    return set(BEFORE)


@pytest.fixture
def after():
    return set(AFTER)


@pytest.fixture
def graph(before):
    return BiTemporalGraph.from_triples(before, valid_from=0)
