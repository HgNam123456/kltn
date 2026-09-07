from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Triple = tuple[str, str, str]


class OpKind(str, Enum):
    ADD = "ADD"
    INVALIDATE = "INVALIDATE"


@dataclass(frozen=True)
class Op:
    """Một thao tác thay đổi trên KG. Provenance không tham gia so sánh/hash."""

    kind: OpKind
    h: str
    r: str
    t: str
    source: str = field(default="", compare=False)
    quote: str = field(default="", compare=False)
    conf: float = field(default=1.0, compare=False)

    @property
    def triple(self) -> Triple:
        return (self.h, self.r, self.t)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value, "h": self.h, "r": self.r, "t": self.t,
            "source": self.source, "quote": self.quote, "conf": self.conf,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Op":
        return cls(
            OpKind(d["kind"]), d["h"], d["r"], d["t"],
            source=d.get("source", ""), quote=d.get("quote", ""), conf=float(d.get("conf", 1.0)),
        )
