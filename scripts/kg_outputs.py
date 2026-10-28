"""Đọc output của các hệ trên mọi benchmark về một dạng chung cho các thước đo không cần gold (độ chính xác theo văn bản
nguồn — scripts/judge_support.py; trùng lặp — scripts/kg_redundancy.py): list văn bản {"id", "text", "triples"} với
triple = [head, relation, tail] (5-tuple COVID bỏ thời gian).

Spec một hệ: tên=loại:đối_số
- covid:<pred.jsonl>          dòng {"row", "quintuples"} (run_covid_nyt, atom_covid, kggen_covid); văn bản từ --covid-data
- covid-released:<cột calib>  output GPT-4.1 tác giả phát hành trong dataset (vd. quintuples_gpt41_from_factoids)
- mine:<graph.jsonl>          dòng {"id", "graph"} (run_mine, kggen_mine); bài luận từ --mine-data (parquet HF)
- mine-released:<cột>         KG phát hành trong dataset MINE (kggen / graphrag_kg / openie_kg), ghép lại đúng bài
                              bằng score_mine.realign (dataset lệch hàng)
- lines:<canon_kg.txt>        định dạng EDC, một văn bản mỗi dòng; văn bản từ --lines-input
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def _triples(items) -> list[list[str]]:
    out = []
    for q in items or []:
        q = list(q)
        if len(q) >= 3 and all(str(x).strip() for x in q[:3]):
            t = [str(x).strip() for x in q[:3]]
            if t not in out:
                out.append(t)
    return out


class Sources:
    """Văn bản nguồn, nạp lười theo loại."""

    def __init__(self, covid_data=None, mine_data=None, lines_input=None):
        self.covid_data, self.mine_data, self.lines_input = covid_data, mine_data, lines_input
        self._covid = self._mine = self._lines = None

    def covid(self) -> list[dict]:
        if self._covid is None:
            self._covid = _jsonl(self.covid_data)
        return self._covid

    def mine(self) -> list[dict]:
        if self._mine is None:
            import pandas as pd

            self._mine = [r for r in pd.read_parquet(self.mine_data).to_dict("records")]
        return self._mine

    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = Path(self.lines_input).read_text(encoding="utf8").splitlines()
        return self._lines


def load(spec: str, src: Sources) -> tuple[str, list[dict]]:
    name, rest = spec.split("=", 1)
    kind, arg = rest.split(":", 1)
    if kind == "covid":
        preds = {d["row"]: d["quintuples"] for d in _jsonl(arg)}
        docs = [{"id": r["row"], "text": r["text"], "triples": _triples(preds.get(r["row"]))} for r in src.covid()]
    elif kind == "covid-released":
        docs = [{"id": r["row"], "text": r["text"], "triples": _triples(r["calib"].get(arg))} for r in src.covid()]
    elif kind == "mine":
        graphs = {int(d["id"]): d.get("graph") or {} for d in _jsonl(arg)}
        docs = [{"id": int(r["id"]), "text": r["essay_content"],
                 "triples": _triples(graphs.get(int(r["id"]), {}).get("relations"))}
                for r in src.mine() if r["essay_content"]]
    elif kind == "mine-released":
        sm = _script("score_mine")
        resp = {"kggen": "kggen_responses"}.get(arg, arg.replace("_kg", "") + "_responses")
        rows = [{**r, arg: None if r[arg] is None else {"entities": [str(e) for e in r[arg]["entities"]],
                                                        "relations": [list(map(str, x)) for x in r[arg]["relations"]]}}
                for r in src.mine()]
        graphs, _ = sm.realign(rows, arg, resp)
        docs = [{"id": int(r["id"]), "text": r["essay_content"], "triples": _triples(graphs[int(r["id"])]["relations"])}
                for r in src.mine() if r["essay_content"] and int(r["id"]) in graphs]
    elif kind == "lines":
        preds = []
        for line in Path(arg).read_text(encoding="utf8").splitlines():
            try:
                preds.append(_triples(ast.literal_eval(line.strip()) if line.strip() else []))
            except Exception:
                preds.append([])
        docs = [{"id": i, "text": t, "triples": preds[i] if i < len(preds) else []} for i, t in enumerate(src.lines())]
    else:
        raise ValueError(f"loại không biết: {kind}")
    return name, docs
