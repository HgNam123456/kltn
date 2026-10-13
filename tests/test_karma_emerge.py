import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KARMA_SRC = ROOT / "tools" / "karma" / "karma_pipeline.py"     # clone YuxingLu613/KARMA @ 82d68ba (gitignored)

spec = importlib.util.spec_from_file_location("karma_emerge", ROOT / "scripts" / "karma_emerge.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_patch_source_replaces_every_occurrence():
    src = "a biomedical x; another biomedical y"
    assert mod.patch_source(src, [("biomedical", "general")]) == "a general x; another general y"


def test_patch_source_fails_loudly_when_upstream_drifted():
    with pytest.raises(ValueError, match="không khớp"):
        mod.patch_source("nothing here", [("biomedical", "general")])


@pytest.mark.skipif(not KARMA_SRC.exists(), reason="cần clone KARMA vào tools/karma")
def test_general_patch_matches_upstream_and_removes_domain_filters():
    src = KARMA_SRC.read_text(encoding="utf8")
    patched = mod.patch_source(src, mod.GENERAL_PATCH)
    compile(patched, str(KARMA_SRC), "exec")
    assert "new biomedical knowledge" not in patched          # Reader không còn lọc theo y sinh
    assert "Extract biomedical entities" not in patched       # NER không còn chỉ tìm bệnh/thuốc/gene
    assert "How important this relationship is in the biomedical domain" not in patched   # relevance từng triple


@pytest.mark.skipif(not KARMA_SRC.exists(), reason="cần clone KARMA vào tools/karma")
def test_load_karma_without_pypdf2_installed():
    karma = mod.load_karma(KARMA_SRC, "general")
    assert hasattr(karma, "KARMA") and hasattr(karma, "KnowledgeTriple")


def test_to_prediction_uses_kg_gen_format():
    pred = mod.to_prediction([("Messi", "member of sports team", "Inter")], "karma/x")
    assert pred["model"] == "karma/x" and pred["model_type"] == "kg-gen"
    assert pred["predicted_triples"] == [{"extracted_relation": ["Messi", "member of sports team", "Inter"],
                                          "triple_qids": [None, None, None],
                                          "triple_labels": [None, None, None]}]


class _Triple:
    def __init__(self, h, r, t):
        self.head, self.relation, self.tail = h, r, t


class _FakeKarma:
    def __init__(self, api_key, base_url, model_name):
        assert base_url == "http://x/v1" and model_name == "local"
        self.intermediate = type("I", (), {"relevant_segments": ["s"], "relationships": [1, 2, 3]})()

    def process_document(self, text):
        assert text == "Messi joined Inter."
        return [_Triple("Messi", "member of sports team", "Inter")]


def test_run_one_records_triples_and_pipeline_counts():
    rec = mod.run_one(_FakeKarma, {"hash_id": "h1", "passage": "Messi joined Inter."}, "http://x/v1", "local")
    assert rec["hash_id"] == "h1"
    assert rec["triples"] == [("Messi", "member of sports team", "Inter")]
    assert rec["n_relevant_segments"] == 1 and rec["n_relationships"] == 3 and rec["seconds"] >= 0
