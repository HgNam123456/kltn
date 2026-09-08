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


class _FakeCompletionsSchemaThenObject:
    """Rejects the first json_schema request with BadRequestError, succeeds otherwise."""

    def __init__(self):
        self.kwargs = []
        self._raised = False

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema" and not self._raised:
            self._raised = True
            import httpx2
            from openai import BadRequestError
            raise BadRequestError(
                message="bad", response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")), body=None,
            )
        content = '{"answer": 7}'
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def test_openai_client_falls_back_to_json_object_once():
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m", max_retries=2)
    c._completions = _FakeCompletionsSchemaThenObject()

    assert c.complete_json("sys", "user", Out).answer == 7
    assert c.complete_json("sys", "user", Out).answer == 7

    kwargs = c._completions.kwargs
    assert len(kwargs) == 3  # schema-fail, object, object
    assert kwargs[0]["response_format"]["type"] == "json_schema"
    assert kwargs[1]["response_format"]["type"] == "json_object"
    assert kwargs[2]["response_format"]["type"] == "json_object"  # 2nd call: never retries json_schema


class _FakeCompletionsRaisesNonBadRequest:
    def __init__(self):
        self.n_calls = 0

    def create(self, **kwargs):
        self.n_calls += 1
        raise RuntimeError("boom")


def test_openai_client_propagates_non_400_errors():
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m", max_retries=2)
    fake = _FakeCompletionsRaisesNonBadRequest()
    c._completions = fake

    with pytest.raises(RuntimeError):
        c.complete_json("sys", "user", Out)

    assert fake.n_calls == 1  # no fallback attempted for a non-400 error


def _bad_request():
    import httpx2 as httpx
    from openai import BadRequestError
    return BadRequestError(message="bad", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)


class _AlwaysBadCompletions:
    def __init__(self):
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        raise _bad_request()


def test_openai_client_keeps_json_schema_when_both_formats_get_400():
    from openai import BadRequestError
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m", max_retries=2)
    c._completions = _AlwaysBadCompletions()
    with pytest.raises(BadRequestError):
        c.complete_json("sys", "user", Out)
    assert c._use_json_schema is True                       # không phải lỗi "không hỗ trợ json_schema"
    assert [k["response_format"]["type"] for k in c._completions.kwargs] == ["json_schema", "json_object"]
