from kgu.extract.text_kg import TemporalKG, extract_schema
from kgu.llm import FakeLLMClient

RELS = {"birthPlace": "Where the subject was born.", "leader": "", "country": "Country of the subject."}


def test_extract_schema_keeps_schema_relations_case_insensitively_and_drops_rest():
    llm = FakeLLMClient([{"facts": [{"subject": "Ab", "relation": "birthPlace", "object": "X"},
                                    {"subject": "Ab", "relation": "country", "object": "Ab"},
                                    {"subject": "Ab", "relation": "birthPlace", "object": "X"}]}])
    assert extract_schema("Ab was born in X.", RELS, llm) == [("Ab", "birthPlace", "X")]


def test_extract_schema_drops_relations_outside_schema_on_unconstrained_path():
    class Loose(FakeLLMClient):                       # đường json_object: server không ép enum
        def complete_json(self, system, user, schema):
            self.calls.append((system, user))
            return type("O", (), {"facts": [type("F", (), {"subject": "Ab", "relation": r, "object": "X"})
                                            for r in ("BIRTHPLACE", "spouse")]})()
    assert extract_schema("t", list(RELS), Loose([])) == [("Ab", "birthPlace", "X")]


def test_extract_schema_prompt_lists_every_relation_with_description():
    llm = FakeLLMClient([{"facts": []}])
    extract_schema("t", RELS, llm, examples="Example 1: ...")
    user = llm.calls[0][1]
    assert all(r in user for r in RELS) and "Where the subject was born." in user
    assert user.index("Example 1") < user.index("Text: t")       # phần tĩnh trước, câu cần trích ở cuối


def test_temporal_kg_reuses_stored_facts_and_closes_ended_ones():
    kg = TemporalKG()
    first = FakeLLMClient([{"facts": [{"head": "Virus", "relation": "occurs_in", "tail": "China",
                                       "valid_from": "2020-01-24", "valid_until": ""}], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", first) == [("Virus", "occurs_in", "China", ["2020-01-24"], [])]
    second = FakeLLMClient([{"facts": [], "ended": [0]}])
    kg.update("The Virus is gone from China.", "2020-03-01", second)
    assert "[0] Virus | occurs_in | China" in second.calls[0][1]          # fact cũ có thực thể được nhắc → STORED
    assert kg.facts[0][4] == ["2020-03-01"]                              # bị báo ended → đóng t_end


def test_temporal_kg_stored_needs_whole_word_and_is_capped():
    kg = TemporalKG(max_stored=2)
    kg.facts = [["A", "r", f"B{i}", [], []] for i in range(5)] + [["US", "r2", "C", [], []]]
    llm = FakeLLMClient([{"facts": [], "ended": [5, -1]}])
    kg.update("A virus did things.", "2020-01-01", llm)
    assert llm.calls[0][1].count("| r |") == 2 and "US" not in llm.calls[0][1].split("RELATIONS")[0]
    assert all(not f[4] for f in kg.facts)                               # số ngoài khối STORED bị bỏ qua


def test_temporal_kg_drops_self_loops_and_duplicates():
    kg = TemporalKG()
    f = {"head": "Virus", "relation": "r", "tail": "virus", "valid_from": "", "valid_until": ""}
    g = {"head": "Virus", "relation": "r", "tail": "China", "valid_from": "", "valid_until": ""}
    assert kg.update("t", "2020-01-01", FakeLLMClient([{"facts": [f, g, g], "ended": []}])) == [("Virus", "r", "China", [], [])]
    assert len(kg.facts) == 1
