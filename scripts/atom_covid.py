"""Baseline ATOM (Lairgi et al., EACL 2026; AuvaLab/itext2kg 1.1.0) trên 2020-COVID-NYT với Gemma 4 qua llama-server,
làm y như script đánh giá của họ (evaluation/atom/exhaustivity/):
- --mode atom: bước 1 factoids_extraction_nyt.py — AtomicFact trên đoạn tin (lead_paragraph_observation_date);
  bước 2 quintuples_extraction_nyt_from_factoids.py — RelationshipsExtractor trên TỪNG factoid (mỗi factoid 1 lệnh
  gọi), system_query = Prompt.temporal_system_query(ngày) + Prompt.EXAMPLES. (Paper dùng factoid của Claude Sonnet 4
  + quintuple của GPT-4.1; ở đây cả hai bước là Gemma.)
- --mode direct: quintuples_extraction_nyt.py — RelationshipsExtractor trên cả đoạn.
Vá để chạy với llama-server (lớp ChatOpenAIJSON, tên giữ chữ "OpenAI" để itext2kg chọn cấu hình provider OpenAI):
- with_structured_output ép method="json_schema" (mặc định function_calling với tool_choice dạng dict, llama-server
  không ép được — itext2kg issue #44);
- đưa JSON schema (kèm mô tả field) vào prompt bằng một system message (schema_hint): OpenAI tự đưa schema của
  response_format vào ngữ cảnh model, còn llama-server chỉ dùng schema để ép ngữ pháp → không vá thì Gemma không thấy
  hướng dẫn chính của ATOM (cách tách factoid, luật quy đổi ngày nằm trong mô tả field). Pipeline của mình
  (kgu.llm.OpenAICompatClient) cũng đưa schema vào prompt như vậy.
Embeddings giả (bước trích không gọi embedding). Không khử trùng quintuple (như cột GPT-4.1 phát hành).
return_exceptions=True (tham số có sẵn của itext2kg): một lệnh gọi trả JSON hỏng chỉ mất factoid đó; mặc định False
làm cả lô (mọi factoid của nhóm ngày) chạy lại 3 lần rồi mất trắng.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/atom_covid.py --mode atom
      --data data/covid_nyt/2020_covid_nyt.jsonl --out out/atom.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dl(v) -> list[str]:
    vals = v if isinstance(v, (list, tuple)) else [v]
    return [str(x).strip() for x in vals if x is not None and str(x).strip() and str(x).strip().lower() != "none"]


def to_quintuple(rel) -> tuple | None:
    h = getattr(getattr(rel, "startNode", None), "name", None)
    t = getattr(getattr(rel, "endNode", None), "name", None)
    r = getattr(rel, "name", None)
    if not (h and r and t):
        return None
    return (str(h).strip(), str(r).strip(), str(t).strip(), _dl(getattr(rel, "t_start", None)),
            _dl(getattr(rel, "t_end", None)))


def flatten(outputs) -> list[tuple]:
    """Danh sách RelationshipsExtractor (None/Exception nếu lệnh gọi hỏng) → 5-tuple, giữ thứ tự, không khử trùng."""
    out = []
    for o in outputs or []:
        for rel in getattr(o, "relationships", None) or []:
            q = to_quintuple(rel) if rel else None
            if q:
                out.append(q)
    return out


def schema_hint(schema) -> str:
    return ("Answer with ONLY a JSON object that follows this JSON schema; the field descriptions are instructions:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False))


def make_parser():
    from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    class ChatOpenAIJSON(ChatOpenAI):
        def with_structured_output(self, schema=None, **kw):
            kw["method"] = "json_schema"
            hint = schema_hint(schema)
            to_msgs = RunnableLambda(lambda p: [SystemMessage(content=hint), HumanMessage(content=str(p))])
            return to_msgs | super().with_structured_output(schema, **kw)

    llm = ChatOpenAIJSON(base_url=os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                         api_key=os.environ.get("KGU_LLM_API_KEY", "none"), model="local", temperature=0,
                         max_retries=2, timeout=600)
    return LangchainOutputParser(llm_model=llm, embeddings_model=FakeEmbeddings(size=8))


def main() -> None:
    from itext2kg.atom.models import AtomicFact, Prompt, RelationshipsExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["atom", "direct"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8, help="số đoạn gửi cùng lúc ở bước factoid")
    args = ap.parse_args()
    rows = sorted(_script("score_covid_nyt").load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    parser, loop = make_parser(), asyncio.new_event_loop()       # một event loop cho cả lượt (client async dùng lại)
    run = loop.run_until_complete

    def quintuples(contexts: list[str], date: str) -> list[tuple]:
        return flatten(run(parser.extract_information_as_json_for_context(
            output_data_structure=RelationshipsExtractor, contexts=contexts,
            system_query=Prompt.temporal_system_query(obs_timestamp=date) + Prompt.EXAMPLES.value,
            return_exceptions=True)))

    if args.mode == "direct":
        step = lambda text, date: quintuples([text], date)                                  # noqa: E731
    else:
        factoids: dict[str, list[str]] = {}
        for i in range(0, len(rows), args.chunk):                     # bước 1: factoid, gửi song song theo cụm
            texts = [r["text"] for r in rows[i:i + args.chunk]]
            try:
                outs = run(parser.extract_information_as_json_for_context(AtomicFact, texts, return_exceptions=True))
            except Exception as e:
                print(f"!!! factoids {i}: {e!r}"[:300], flush=True)
                outs = []
            for j, text in enumerate(texts):
                o = outs[j] if j < len(outs) else None
                factoids[text] = [str(f) for f in (getattr(o, "atomic_fact", None) or []) if str(f).strip()]
            print(f"factoids {min(i + args.chunk, len(rows))}/{len(rows)}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}.factoids.json").write_text(
            json.dumps([{"row": r["row"], "factoids": factoids[r["text"]]} for r in rows], ensure_ascii=False),
            encoding="utf8")
        step = lambda text, date: quintuples(factoids[text], date) if factoids[text] else []  # noqa: E731
    _script("run_covid_nyt").run_rows(rows, step, args.out)
    print(f"{args.mode}: xong {len(rows)} nhóm ngày → {args.out}", flush=True)


if __name__ == "__main__":
    main()
