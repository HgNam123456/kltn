from kgu.data.nba import Example
from kgu.extract.llm import LLMExtractor
from kgu.graph import BiTemporalGraph
from kgu.llm import FakeLLMClient
from kgu.types import Op, OpKind
from tests.conftest import PLAYS, TEAM


def _ex(before, after):
    return Example(0, "trade", "2021", "Messi leaves Barca and joins Inter.", ["Messi", "Barca", "Inter"], before, after)


def test_extractor_parses_and_filters(before, after):
    client = FakeLLMClient([{"ops": [
        {"kind": "INVALIDATE", "h": "Messi", "r": PLAYS, "t": "Barca", "quote": "leaves Barca"},
        {"kind": "ADD", "h": "Messi", "r": PLAYS, "t": "Inter", "quote": "joins Inter"},
        {"kind": "ADD", "h": "Messi", "r": PLAYS, "t": "Mars FC"},          # entity lạ → bỏ
        {"kind": "INVALIDATE", "h": "Messi", "r": PLAYS, "t": "Inter"},     # không active → bỏ
        {"kind": "ADD", "h": "Pedri", "r": PLAYS, "t": "Barca"},            # đã active → bỏ
    ]}])
    ext = LLMExtractor(client, relations=[PLAYS, TEAM])
    ops = ext.extract(_ex(before, after), BiTemporalGraph.from_triples(before), 0)
    assert ops == [
        Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
        Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
    ]
    assert ops[0].source == "llm" and ops[0].quote == "leaves Barca"
    assert ext.n_dropped == 3 and ext.n_calls == 1


def test_prompt_contains_text_relations_and_mentioned_edges(before, after):
    client = FakeLLMClient([{"ops": []}])
    LLMExtractor(client, relations=[PLAYS, TEAM]).extract(_ex(before, after), BiTemporalGraph.from_triples(before), 0)
    system, user = client.calls[0]
    assert "Messi leaves Barca" in user
    assert PLAYS in user and TEAM in user
    assert "(Messi, plays_for, Barca)" in user
    assert "ADD" in system and "INVALIDATE" in system


import json


def test_schema_restricts_relation_to_enum():
    ext = LLMExtractor(FakeLLMClient([]), relations=[PLAYS, TEAM])
    s = json.dumps(ext.schema.model_json_schema())
    assert f'"enum": ["{PLAYS}", "{TEAM}"]' in s


def test_prompt_has_few_shot_and_direction_rule(before, after):
    client = FakeLLMClient([{"ops": []}])
    LLMExtractor(client, relations=[PLAYS, TEAM]).extract(_ex(before, after), BiTemporalGraph.from_triples(before), 0)
    system, user = client.calls[0]
    assert "VÍ DỤ" in user and '"kind": "INVALIDATE"' in user
    assert "cùng chiều" in system
