from kgu.extract.condense import condense, missing_items
from kgu.llm import FakeLLMClient

TEXT = "Italy reported 1,694 cases on Sunday, most of them in Lombardy."


def test_missing_items_compares_numbers_by_digits_and_names_case_insensitively():
    assert missing_items(TEXT, ["Italy reported 1694 cases on sunday."]) == ["Lombardy"]


def test_missing_items_skips_leading_function_words():
    assert missing_items("The World Health Organization said it. But nothing else.",
                         ["World Health Organization said something."]) == []


def test_condense_asks_once_more_for_missing_items():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday."]},
                         {"sentences": ["Most of the 1,694 cases are in Lombardy.",
                                        "Italy reported 1,694 cases on Sunday."]}])
    sents, stats = condense(TEXT, llm)
    assert sents == ["Italy reported 1,694 cases on Sunday.", "Most of the 1,694 cases are in Lombardy."]
    assert stats == {"sentences": 2, "missing_before": 1, "missing_after": 0}
    assert llm.calls[0][1] == f"Passage: {TEXT}" and "ITEMS: Lombardy" in llm.calls[1][1]


def test_condense_makes_one_call_when_nothing_is_missing():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday.", "Most cases are in Lombardy.", " "]}])
    sents, stats = condense(TEXT, llm)
    assert llm.n_calls == 1 and sents == ["Italy reported 1,694 cases on Sunday.", "Most cases are in Lombardy."]


def test_condense_keeps_sentences_when_fill_call_fails():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday."]}])   # lệnh bổ sung: hết response → lỗi
    sents, stats = condense(TEXT, llm)
    assert sents == ["Italy reported 1,694 cases on Sunday."]
    assert stats["missing_after"] == 1 and "fill_error" in stats
