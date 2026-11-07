"""Định dạng KG cho trước (header base + chênh lệch) của kgu.data.jsonl."""
import json

from kgu.data.jsonl import load_jsonl


def _write(tmp_path, lines):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf8")
    return p


def test_given_kg_non_cumulative(tmp_path):
    base = [["a", "r1", "b"], ["b", "r2", "c"], ["x", "r3", "y"]]
    p = _write(tmp_path, [
        {"base": base, "cumulative": False},
        {"idx": 0, "mentioned": ["a", "c"], "before_del": [["a", "r1", "b"]],
         "after_add": [["a", "r1", "b"], ["a", "r2", "c"]]},
        {"idx": 1, "mentioned": ["x", "y"], "after_del": [["x", "r3", "y"]]},
    ])
    ex, rels, extras = load_jsonl(p)
    assert len(ex) == 2 and rels == ["r1", "r2", "r3"] and extras == [{}, {}]
    assert ex[0].before == {("b", "r2", "c"), ("x", "r3", "y")}
    assert ex[0].after == {("a", "r1", "b"), ("b", "r2", "c"), ("x", "r3", "y"), ("a", "r2", "c")}
    assert ex[1].before == {tuple(t) for t in base}            # không tích lũy: bài 2 lại xuất phát từ base
    assert ex[1].after == {("a", "r1", "b"), ("b", "r2", "c")}


def test_given_kg_cumulative_and_closed_relations(tmp_path):
    p = _write(tmp_path, [
        {"base": [["a", "r1", "b"]], "cumulative": True, "relations": ["r1", "r9"]},
        {"idx": 0, "after_add": [["a", "r1", "c"]]},
        {"idx": 1, "before_add": [["q", "r1", "z"]], "after_del": [["a", "r1", "b"]]},
    ])
    ex, rels, _ = load_jsonl(p)
    assert rels == ["r1", "r9"]                                 # header quyết định tập quan hệ đóng
    assert ex[1].before == {("a", "r1", "b"), ("a", "r1", "c"), ("q", "r1", "z")}
    assert ex[1].after == {("a", "r1", "c"), ("q", "r1", "z")}
    assert [e.idx for e in ex[0:2]] == [0, 1]
    ex2, _, _ = load_jsonl(p, limit=1)
    assert len(ex2) == 1


def test_slice_format_unchanged(tmp_path):
    p = _write(tmp_path, [{"idx": 0, "before": [["a", "r", "b"]], "after": [["a", "r", "b"], ["a", "s", "c"]],
                           "extra": {"k": 1}}])
    ex, rels, extras = load_jsonl(p)
    assert ex[0].after == {("a", "r", "b"), ("a", "s", "c")} and rels == ["r", "s"] and extras == [{"k": 1}]
