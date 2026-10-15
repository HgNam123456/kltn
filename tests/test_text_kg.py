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
    assert kg.update("The Virus is gone from China.", "2020-03-01", second) == [
        ("Virus", "occurs_in", "China", ["2020-01-24"], ["2020-03-01"])]      # fact bị đóng cũng là fact của đoạn
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


def _f(head, rel, tail, sentence):
    return {"head": head, "relation": rel, "tail": tail, "valid_from": "", "valid_until": "", "sentence": sentence}


def test_condense_mode_extracts_on_numbered_sentences_and_refills_uncovered_ones():
    kg = TemporalKG(condense=True)
    kg.facts = [["Italy", "has_status", "Low Case Count", [], []]]
    llm = FakeLLMClient([
        {"sentences": ["Italy reports 5 cases.", "Lombardy is in Italy.", "Milan is in Lombardy."]},
        {"facts": [_f("Italy", "reports", "5 Cases", 1), _f("Rome", "is_in", "Italy", 99)], "ended": []},
        {"facts": [_f("Lombardy", "is_located_in", "Italy", 2), _f("Milan", "is_located_in", "Lombardy", 3)],
         "ended": [0]},
    ])
    out = kg.update("Italy reports 5 cases in Lombardy, Italy, and Milan.", "2020-03-02", llm)
    assert out == [("Italy", "reports", "5 Cases", [], []), ("Rome", "is_in", "Italy", [], []),
                   ("Lombardy", "is_located_in", "Italy", [], []), ("Milan", "is_located_in", "Lombardy", [], [])]
    assert "[1] Italy reports 5 cases.\n[2] Lombardy is in Italy.\n[3] Milan is in Lombardy." in llm.calls[1][1]
    assert "[2] Lombardy is in Italy.\n[3] Milan" in llm.calls[2][1] and "[1] Italy" not in llm.calls[2][1]
    assert kg.facts[0][4] == []                       # "ended" của lệnh bổ sung bị bỏ qua
    assert kg.last_stats == {"sentences": 3, "missing_before": 0, "missing_after": 0, "refill_sentences": 2}


def test_condense_mode_falls_back_to_passage_when_condense_gives_nothing():
    kg = TemporalKG(condense=True)
    llm = FakeLLMClient([{"sentences": []},
                         {"facts": [{"head": "Virus", "relation": "occurs_in", "tail": "China",
                                     "valid_from": "", "valid_until": ""}], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", llm) == [("Virus", "occurs_in", "China", [], [])]
    assert "Passage: Virus spreads in China." in llm.calls[1][1] and "SENTENCES" not in llm.calls[1][1]


def test_condense_mode_falls_back_to_passage_when_condense_output_is_broken():
    kg = TemporalKG(condense=True)
    llm = FakeLLMClient([{"sentence": "truncated"}, {"facts": [], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", llm) == []
    assert "condense_error" in kg.last_stats and "Passage: Virus spreads in China." in llm.calls[1][1]


def test_stored_matches_passage_through_aliases():
    kg = TemporalKG()
    kg.facts = [["World Health Organization", "warns", "Countries", [], []]]
    kg.aliases = {"World Health Organization": {"WHO"}}
    llm = FakeLLMClient([{"facts": [], "ended": []}])
    kg.update("WHO said the risk is high.", "2020-03-01", llm)
    assert "[0] World Health Organization | warns | Countries" in llm.calls[0][1]


def test_condense_mode_drops_the_observation_date_header_before_condensing():
    kg = TemporalKG(condense=True)
    llm = FakeLLMClient([{"sentences": ["Virus spreads in China."]}, {"facts": [], "ended": []},
                         {"facts": [], "ended": []}])
    kg.update("Observation date: 2020-01-28\nVirus spreads in China.", "2020-01-28", llm)
    assert "Virus spreads in China." in llm.calls[0][1]
    assert "Observation date" not in llm.calls[0][1]


def test_new_facts_are_stored_under_the_canonical_name_of_an_alias():
    kg = TemporalKG()
    kg.aliases = {"World Health Organization": {"WHO"}}
    llm = FakeLLMClient([{"facts": [{"head": "WHO", "relation": "warns", "tail": "Countries",
                                     "valid_from": "", "valid_until": ""}], "ended": []}])
    out = kg.update("WHO warns countries.", "2020-03-01", llm)
    assert out == [("WHO", "warns", "Countries", [], [])]
    assert kg.facts == [["World Health Organization", "warns", "Countries", [], []]]
