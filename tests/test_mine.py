import importlib.util
from pathlib import Path


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


run_mine = _script("run_mine")
score_mine = _script("score_mine")


def test_paragraphs_strip_fences_and_merge_title_into_next_paragraph():
    essay = "```The Life Cycle of a Butterfly\n\nButterflies are insects. They fly.\n\n\nEggs hatch into larvae.```"
    assert run_mine.paragraphs(essay) == ["The Life Cycle of a Butterfly\nButterflies are insects. They fly.",
                                          "Eggs hatch into larvae."]


def test_paragraphs_of_empty_essay_is_empty():
    assert run_mine.paragraphs("") == []
    assert run_mine.paragraphs("```\n\n```") == []


def test_graph_dict_uses_readable_relations_and_drops_duplicates_and_self_loops():
    facts = [["Butterfly", "lays_eggs_on", "host plant", [], []],
             ["Butterfly", "lays_eggs_on", "host plant", ["2020-01-01"], []],
             ["larva", "becomes", "larva", [], []],
             ["larva", "becomes", "pupa", [], []]]
    g = run_mine.graph_dict(facts)
    assert g["relations"] == [["Butterfly", "lays eggs on", "host plant"], ["larva", "becomes", "pupa"]]
    assert g["entities"] == ["Butterfly", "host plant", "larva", "pupa"]
    assert g["edges"] == ["becomes", "lays eggs on"]


def test_verdict_parses_ints_strings_and_falls_back_to_zero():
    assert score_mine.verdict(1) == 1
    assert score_mine.verdict("1") == 1
    assert score_mine.verdict(" 0 ") == 0
    assert score_mine.verdict("yes") == 0
    assert score_mine.verdict(None) == 0


def test_accuracy_is_mean_of_per_essay_accuracy():
    per_essay = {0: [1, 1, 0, 0], 1: [1, 1, 1, 1]}
    assert score_mine.accuracy(per_essay) == {"essays": 2, "queries": 8, "accuracy": 0.75, "micro": 0.75}
    assert score_mine.accuracy({}) == {"essays": 0, "queries": 0, "accuracy": 0.0, "micro": 0.0}


def test_released_accuracy_parses_percent_strings_and_skips_missing():
    assert score_mine.released_accuracy(["53.33%", None, "100.00%", ""]) == {"essays": 2, "accuracy": 0.76665}
