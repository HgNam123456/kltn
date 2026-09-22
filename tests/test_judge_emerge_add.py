from kgu.data.emerge import OPS, EmergeExample, Mention
from kgu.judge.emerge_add import allowed_relations, index_relations, judge_add, render
from kgu.llm import FakeLLMClient

KG = {("Q1", "P54", "Q2"), ("Q9", "P17", "Q4"), ("Q2", "P17", "Q4")}
LABELS = {"Q1": "Messi", "Q2": "Barca", "Q3": "Inter", "Q4": "Spain", "P54": "team", "P17": "country"}


def _ex():
    ms = [Mention(q, q, 0, 1) for q in ("Q1", "Q2", "Q3")]
    empty = {op: set() for op in OPS}
    return EmergeExample("h", "2019-01-01", "2019-01-08", "Messi", "Messi joined Inter.", ms, dict(empty), dict(empty))


def test_allowed_relations_from_1hop_of_mentions():
    assert allowed_relations(_ex(), index_relations(KG)) == ["P17", "P54"]     # P17 qua Barca–Spain


def test_render_numbers_entities_and_lists_stored():
    text = render(_ex(), [("Q1", "P54", "Q2")], ["P17", "P54"], LABELS)
    assert "[0] Messi  [1] Barca  [2] Inter" in text
    assert "STORED: [0] team [1]" in text and "RELATIONS: country, team" in text
    text = render(_ex(), [], ["P17"], {"P17": "country"})          # thiếu nhãn KG -> dùng chữ trong passage
    assert "[0] Q1  [1] Q2  [2] Q3" in text


def test_judge_add_maps_labels_and_filters():
    llm = FakeLLMClient([{"facts": [
        {"head": 0, "relation": "Team", "tail": 2},        # hợp lệ, nhãn khác hoa thường
        {"head": 0, "relation": "team", "tail": 1},        # đã có trong graph -> bỏ
        {"head": 0, "relation": "spouse", "tail": 2},      # quan hệ không trong danh sách -> bỏ
        {"head": 0, "relation": "team", "tail": 7},        # chỉ số ngoài danh sách -> bỏ
        {"head": 1, "relation": "team", "tail": 1},        # tự nối -> bỏ
    ]}])
    out = judge_add(_ex(), [("Q1", "P54", "Q2")], ["P17", "P54"], LABELS, llm)
    assert out == {("Q1", "P54", "Q3")}


def test_judge_add_skips_llm_when_nothing_to_ask():
    llm = FakeLLMClient([])
    assert judge_add(_ex(), [], [], LABELS, llm) == set()
    assert llm.n_calls == 0
