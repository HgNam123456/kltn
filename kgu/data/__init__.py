from __future__ import annotations

from dataclasses import dataclass

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
