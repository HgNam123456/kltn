"""Kiểm tra server LLM local: gọi /v1/chat/completions với json_schema, đo tốc độ.

Chạy sau khi scripts/llm_server.ps1 đã lên:  python scripts/llm_smoke.py [base_url]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252, không in được tiếng Việt
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["ADD", "INVALIDATE"]},
                    "h": {"type": "string"}, "r": {"type": "string"}, "t": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["kind", "h", "r", "t", "quote"],
            },
        }
    },
    "required": ["ops"],
}

body = {
    "model": "local",
    "temperature": 0,
    "max_tokens": 400,
    # Qwen3.5 là model có thinking; tắt để trả JSON ngay (server cũng đã tắt bằng --reasoning off)
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content":
            "You extract explicit knowledge-graph changes from a news sentence.\n"
            "Allowed relations: <player>, <teammate>, <head_coach>.\n"
            "Edge format is (h, r, t). INVALIDATE = an existing edge is no longer true. ADD = a new edge is stated.\n"
            "Only use entity names exactly as given. Copy the supporting text into 'quote'. Output JSON only."},
        {"role": "user", "content":
            "Existing edges:\n(Indiana_Pacers, <player>, Thaddeus_Young_2017-18)\n\n"
            "News: the Indiana_Pacers announced friday they have waived forward Thaddeus_Young_2017-18 .\n\n"
            "List the ADD / INVALIDATE ops the news states directly."},
    ],
    "response_format": {"type": "json_schema", "json_schema": {"name": "ops", "schema": SCHEMA}},
}

req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)
dt = time.time() - t0
content = resp["choices"][0]["message"]["content"]
usage = resp.get("usage", {})
print("content:", content)
parsed = json.loads(content)
assert "ops" in parsed, "không đúng schema"
print(f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} "
      f"time={dt:.1f}s  ~{usage.get('completion_tokens', 0) / dt:.1f} tok/s (gồm cả prompt)")
timings = resp.get("timings")
if timings:
    print(f"prompt: {timings.get('prompt_per_second', 0):.1f} tok/s | generate: {timings.get('predicted_per_second', 0):.1f} tok/s")
