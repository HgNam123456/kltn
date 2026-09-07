from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kgu.types import Triple


@dataclass
class Example:
    idx: int
    event: str
    season: str
    text: str
    mentioned: list[str]
    before: set[Triple]
    after: set[Triple]


def load_id_map(path: Path) -> dict[int, str]:
    """Đọc file `name<TAB>id`; bỏ qua dòng đầu (số đếm) và dòng không có TAB."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if "\t" not in line:
            continue
        name, idx = line.rsplit("\t", 1)
        out[int(idx)] = name
    return out


@dataclass
class Vocab:
    entity: dict[int, str]
    relation: dict[int, str]
    token: dict[int, str]

    @classmethod
    def load(cls, data_dir: Path) -> "Vocab":
        return cls(
            entity=load_id_map(data_dir / "entity2id.txt"),
            relation=load_id_map(data_dir / "relation2id.txt"),
            token=load_id_map(data_dir / "token2id.txt"),
        )

    @property
    def relations(self) -> list[str]:
        return [self.relation[k] for k in sorted(self.relation)]

    def triple(self, raw: list[int]) -> Triple:
        h, r, t = raw
        return (self.entity[h], self.relation[r], self.entity[t])

    def _name(self, x, table: dict[int, str]) -> str:
        return table[x] if isinstance(x, int) else str(x)

    def text(self, raw: list) -> str:
        return " ".join(self._name(t, self.token) for t in raw)

    def entities(self, raw: list) -> list[str]:
        return [self._name(e, self.entity) for e in raw]


def load_split(data_dir: Path, split: str, limit: int | None = None) -> list[Example]:
    vocab = Vocab.load(data_dir)
    raw = json.loads((data_dir / f"NBAtransactions_{split}.json").read_text(encoding="utf8"))
    if limit is not None:
        raw = raw[:limit]
    return [
        Example(
            idx=i,
            event=str(d.get("event", "")),
            season=str(d.get("season", "")),
            text=vocab.text(d["text"]),
            mentioned=vocab.entities(d.get("text_mentioned_entities", [])),
            before={vocab.triple(t) for t in d["subgraph_before"]},
            after={vocab.triple(t) for t in d["subgraph_after"]},
        )
        for i, d in enumerate(raw)
    ]
