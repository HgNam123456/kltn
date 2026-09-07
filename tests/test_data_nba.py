import json
from pathlib import Path

from kgu.data.nba import Vocab, load_split


def _write_fixture(d: Path) -> None:
    (d / "entity2id.txt").write_text("3\nBarca\t0\nMessi\t1\nInter\t2\n", encoding="utf8")
    (d / "relation2id.txt").write_text("1\n<player>\t0\n", encoding="utf8")
    (d / "token2id.txt").write_text("3\nmessi\t0\njoins\t1\ninter\t2\n", encoding="utf8")
    ex = {
        "event": "free agency", "season": "2021-22",
        "text": [0, 1, 2],
        "text_mentioned_entities": [1, 2],
        "subgraph_before": [[0, 0, 1]],
        "subgraph_after": [[2, 0, 1]],
    }
    (d / "NBAtransactions_test.json").write_text(json.dumps([ex, ex]), encoding="utf8")


def test_vocab_skips_count_line(tmp_path):
    _write_fixture(tmp_path)
    v = Vocab.load(tmp_path)
    assert v.entity == {0: "Barca", 1: "Messi", 2: "Inter"}
    assert v.relations == ["<player>"]


def test_load_split_decodes_names(tmp_path):
    _write_fixture(tmp_path)
    exs = load_split(tmp_path, "test")
    assert len(exs) == 2
    ex = exs[0]
    assert ex.idx == 0 and ex.event == "free agency"
    assert ex.text == "messi joins inter"
    assert ex.mentioned == ["Messi", "Inter"]
    assert ex.before == {("Barca", "<player>", "Messi")}
    assert ex.after == {("Inter", "<player>", "Messi")}


def test_load_split_limit(tmp_path):
    _write_fixture(tmp_path)
    assert len(load_split(tmp_path, "test", limit=1)) == 1
