from __future__ import annotations

import json
import os
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    pass


class LLMClient(Protocol):
    n_calls: int

    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...


def extract_json(text: str) -> str:
    text = re.sub(r"`{3}(?:json)?", "", text)   # bỏ fence markdown (không viết 3 backtick liền để không phá markdown)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text.strip()


class FakeLLMClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.n_calls = 0

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((system, user))
        self.n_calls += 1
        return schema.model_validate(self.responses.pop(0))


class OpenAICompatClient:
    """Gọi bất kỳ endpoint OpenAI-compatible (vLLM, Ollama, OpenRouter, DashScope)."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 max_retries: int = 3, timeout: float = 120.0) -> None:
        self.model, self.temperature, self.max_retries = model, temperature, max_retries
        self.n_calls = 0
        self._completions = None
        self._init = dict(base_url=base_url, api_key=api_key, timeout=timeout)
        self._use_json_schema = True  # nhớ lại nếu server đã từ chối json_schema, tránh trả 2 round-trip mỗi lần

    @property
    def completions(self):
        if self._completions is None:
            from openai import OpenAI
            self._completions = OpenAI(**self._init).chat.completions
        return self._completions

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        system_full = (
            f"{system}\n\nTrả lời DUY NHẤT một JSON object hợp lệ theo schema sau, không giải thích:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            self.n_calls += 1
            # Qwen3.5 là model có thinking: tắt qua chat_template_kwargs (llama-server đọc extra_body),
            # nếu không nội dung sẽ nằm trong reasoning_content và content rỗng.
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
            if self._use_json_schema:
                from openai import BadRequestError
                try:
                    resp = self.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        response_format={"type": "json_schema",
                                         "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
                        extra_body=extra,
                    )
                except BadRequestError:
                    # Có thể là "không hỗ trợ json_schema" HOẶC lỗi khác (tràn ctx...). Chỉ khi json_object
                    # thành công mới kết luận server không hỗ trợ json_schema và nhớ lại cho lần sau.
                    resp = self.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        response_format={"type": "json_object"}, extra_body=extra,
                    )
                    self._use_json_schema = False
            else:
                resp = self.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    response_format={"type": "json_object"}, extra_body=extra,
                )
            raw = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Output không hợp lệ (error: {str(e)[:500]}). Xuất lại đúng schema, chỉ JSON."})
        raise LLMOutputError(f"invalid JSON after {self.max_retries} tries: {last_err}")


def client_from_env() -> OpenAICompatClient:
    """Mặc định: llama-server local (scripts/llm_server.ps1) trên http://localhost:8080/v1."""
    return OpenAICompatClient(
        base_url=os.environ.get("KGU_LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("KGU_LLM_API_KEY", "none"),
        model=os.environ.get("KGU_LLM_MODEL", "local"),
    )
