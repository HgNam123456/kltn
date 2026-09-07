import pytest
from pydantic import BaseModel

from kgu.llm import FakeLLMClient, LLMOutputError, OpenAICompatClient, extract_json


class Out(BaseModel):
    answer: int


def test_fake_client_returns_in_order_and_records_calls():
    c = FakeLLMClient([{"answer": 1}, {"answer": 2}])
    assert c.complete_json("sys", "u1", Out).answer == 1
    assert c.complete_json("sys", "u2", Out).answer == 2
    assert c.calls == [("sys", "u1"), ("sys", "u2")]
    assert c.n_calls == 2


def test_extract_json_strips_fences_and_prose():
    fence = "`" * 3
    assert extract_json(f'Sure!\n{fence}json\n{{"answer": 3}}\n{fence}\nDone.') == '{"answer": 3}'


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        content = self.contents.pop(0)
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def _client_with(contents):
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m", max_retries=2)
    c._completions = _FakeCompletions(contents)
    return c


def test_openai_client_retries_on_invalid_json_then_succeeds():
    c = _client_with(['{"answer": "not-int"}', '{"answer": 5}'])
    assert c.complete_json("sys", "user", Out).answer == 5
    assert c.n_calls == 2
    assert "not-int" in c._completions.kwargs[1]["messages"][-1]["content"] or "error" in c._completions.kwargs[1]["messages"][-1]["content"].lower()


def test_openai_client_raises_after_retries():
    c = _client_with(["garbage", "still garbage"])
    with pytest.raises(LLMOutputError):
        c.complete_json("sys", "user", Out)
