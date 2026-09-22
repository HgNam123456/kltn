from kgu.data.emerge import OPS, EmergeExample
from kgu.eval.emerge import bootstrap_ci, executable_scores, instance_scores, shipped_predictions


def _ex(hash_id, **gold):
    g = {op: set() for op in OPS}
    g.update({k.replace("_", "-"): v for k, v in gold.items()})
    return EmergeExample(hash_id, "2019-01-01", "2019-01-08", "", "", [], g, {op: set() for op in OPS})


def test_executable_scores_macro_over_instances_with_gold():
    a = _ex("a", d_triples={("Q1", "P54", "Q2"), ("Q1", "P54", "Q3")})
    b = _ex("b", d_triples={("Q4", "P54", "Q5")})
    c = _ex("c")                                           # không có gold Deprecate -> không vào mẫu số
    preds = {
        "a": {"d-triples": {("Q1", "P54", "Q2"), ("Q9", "P1", "Q9")}},   # recall 1/2, precision 1/2
        "c": {"d-triples": {("Q7", "P1", "Q8")}},
    }                                                      # b không dự đoán -> 0
    s = executable_scores([a, b, c], preds)["d-triples"]
    assert s["n"] == 2
    assert s["recall"] == 0.25 and s["precision"] == 0.25


def test_executable_scores_ignores_non_qid_predictions():
    a = _ex("a", e_triples={("Q1", "P54", "Q2")})
    s = executable_scores([a], {"a": {"e-triples": {("Q1", "P54", "Q2"), ("--NME--", "P54", "Q2")}}})
    assert s["e-triples"]["precision"] == 1.0


def test_shipped_predictions_routes_by_operation():
    raw = {"predictions": {"m": {"predicted_triples": [
        {"operation": "DEPRECATE", "triple_qids": ["Q1", "P6", "Q2"]},
        {"operation": "MINT_ADD", "triple_qids": [None, "P6", "Q2"]},
        {"operation": "ADD", "triple_qids": ["Q3", "P6", "Q2"]},
    ]}}}
    p = shipped_predictions(raw, "m")
    assert p["d-triples"] == {("Q1", "P6", "Q2")} and p["e-triples"] == {("Q3", "P6", "Q2")}
    assert p["ee-triples"] == set()


def test_instance_scores_keyed_by_hash():
    a = _ex("a", d_triples={("Q1", "P54", "Q2")})
    rows = instance_scores([a, _ex("c")], {"a": {"d-triples": {("Q1", "P54", "Q2")}}})
    assert rows["d-triples"] == {"a": (1.0, 1.0)}


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(vals, n_boot=500)
    assert lo < 0.5 < hi and hi - lo < 0.3
    assert bootstrap_ci(vals, n_boot=500) == (lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert bootstrap_ci([]) == (0.0, 0.0)
