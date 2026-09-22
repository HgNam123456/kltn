from kgu.data.emerge import OPS, EmergeExample, Mention
from kgu.judge.emerge import candidates, index_out_edges, judge, render, to_ops
from kgu.llm import FakeLLMClient

KG = {
    ("Q1", "P54", "Q2"),      # Messi – team – Barca   (cả hai được nhắc)
    ("Q1", "P27", "Q3"),      # Messi – citizenship – Argentina (Argentina không được nhắc)
    ("Q7", "P54", "Q2"),      # Pedri – team – Barca   (Pedri không được nhắc)
    ("Q2", "P17", "Q4"),      # Barca – country – Spain
}
LABELS = {"Q1": "Messi", "Q2": "Barca", "Q4": "Spain", "P54": "team", "P17": "country"}
A, B = ("Q1", "P54", "Q2"), ("Q2", "P17", "Q4")


def _ex():
    ms = [Mention(q, q, 0, 1) for q in ("Q1", "Q2", "Q4")]
    empty = {op: set() for op in OPS}
    return EmergeExample("h", "2019-01-01", "2019-01-08", "Messi", "Messi left Barca.", ms, dict(empty), dict(empty))


def test_candidates_need_both_ends_mentioned():
    assert candidates(_ex(), index_out_edges(KG)) == [A, B]


def test_render_has_snapshot_date_labels_and_numbers():
    text = render(_ex(), [A, B], LABELS)
    assert "GRAPH DATE: 2019-01-01" in text and "TODAY: 2019-01-08" in text
    assert "[0] Messi | team | Barca" in text and "[1] Barca | country | Spain" in text
    assert "Messi left Barca." in text


def test_judge_batch_two_step_and_missing_is_unrelated():
    llm = FakeLLMClient([{"facts": [
        {"id": 0, "discussed": True, "ended": True},
        {"id": 7, "discussed": True, "ended": False},
    ]}])
    assert judge(_ex(), [A, B], LABELS, llm, batch_size=12) == {A: "ended", B: "unrelated"}
    assert llm.n_calls == 1


def test_judge_single_mode_one_call_per_fact():
    llm = FakeLLMClient([{"discussed": False, "ended": True}, {"discussed": True, "ended": False}])
    assert judge(_ex(), [A, B], LABELS, llm, batch_size=1) == {A: "unrelated", B: "holds"}   # ended cần discussed
    assert llm.n_calls == 2
    assert "[0] Barca | country | Spain" in llm.calls[1][1]


def test_judge_skips_llm_without_candidates():
    llm = FakeLLMClient([])
    assert judge(_ex(), [], LABELS, llm) == {}
    assert llm.n_calls == 0


def test_to_ops_exists_modes():
    v = {A: "ended", B: "unrelated"}
    assert to_ops(v, "all") == {"x-triples": {B}, "d-triples": {A}}
    assert to_ops(v, "llm") == {"x-triples": set(), "d-triples": {A}}
