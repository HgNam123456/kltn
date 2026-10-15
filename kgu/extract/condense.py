"""Rút đoạn tin thành câu đơn trước khi trích (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md
§2.1): mỗi câu một ý, tự đứng được, giữ nguyên số/tên/cụm thời gian (quy đổi ngày vẫn do bước trích làm). Sau lệnh gọi,
số và cụm viết hoa của đoạn mà không câu nào chứa bị coi là sót → một lệnh gọi bổ sung câu cho chúng."""
from __future__ import annotations

import re

from pydantic import BaseModel

CONDENSE_SYSTEM = """Rewrite the news passage as a list of short simple sentences.
- Each sentence states exactly ONE fact or idea.
- Each sentence stands alone: replace pronouns and vague references (he, she, it, they, the agency, the city) with \
the full name they refer to.
- Keep every number, name, place and time expression exactly as written in the passage ("on Sunday", "last week", \
"1,694 cases"); do not convert dates.
- Keep ALL the information: every fact, number, place, cause, status and statement of the passage must appear in some \
sentence. Do not add anything the passage does not say.

Example:
Passage: "Anna, a teacher, joined a course at Hanoi University on Monday, which she said was too short."
-> {"sentences": ["Anna is a teacher.", "Anna joined a course at Hanoi University on Monday.", \
"Anna said the course at Hanoi University was too short."]}"""

FILL_SYSTEM = """You get a news passage, the simple sentences already written from it, and ITEMS of the passage that \
no sentence mentions yet. Write the missing short simple sentences (one fact each, standing alone, numbers, names and \
time expressions kept as written) that state what the passage says about these items. Do not repeat sentences already \
written. Return {"sentences": [...]} with only the new sentences."""

STOP = frozenset("""a an the and or but if so yet as at by for from in into of on onto to with without within about
after before during since until while when where what which who whom whose why how this that these those there here
it its he him his she her they them their we us our you your i me my mr ms mrs dr some many most more all any each
every no not one also just still only even""".split())

_NUM = re.compile(r"\d[\d,.]*\d|\d")
_CAP = re.compile(r"\b[A-Z][\w'’&-]*(?:\s+[A-Z][\w'’&-]*)*")


class Sentences(BaseModel):
    sentences: list[str]


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _clean(sentences: list[str]) -> list[str]:
    out: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and s not in out:
            out.append(s)
    return out


def missing_items(text: str, sentences: list[str], limit: int = 15) -> list[str]:
    """Số (so theo chữ số, "1,694" = "1694") và cụm viết hoa (bỏ từ chức năng ở đầu) của đoạn mà không câu nào chứa
    (không phân biệt hoa thường)."""
    joined = " ".join(sentences)
    low, nums = joined.lower(), {_digits(m) for m in _NUM.findall(joined)}
    out: list[str] = []
    for m in _NUM.findall(text):
        if _digits(m) not in nums and m not in out:
            out.append(m)
    for m in _CAP.findall(text):
        words = m.split()
        while words and words[0].lower() in STOP:
            words = words[1:]
        phrase = " ".join(words)
        if phrase and phrase.lower() not in low and phrase not in out:
            out.append(phrase)
    return out[:limit]


def condense(text: str, llm) -> tuple[list[str], dict]:
    """→ (câu đơn, thống kê). Lệnh gọi đầu lỗi → ném lỗi (TemporalKG quay về v0); lệnh bổ sung lỗi → giữ câu đã có."""
    sents = _clean(llm.complete_json(CONDENSE_SYSTEM, f"Passage: {text}", Sentences).sentences)
    miss = missing_items(text, sents) if sents else []
    stats = {"sentences": len(sents), "missing_before": len(miss), "missing_after": len(miss)}
    if miss:
        user = (f"Passage: {text}\n\nSENTENCES:\n" + "\n".join(f"- {s}" for s in sents)
                + "\n\nITEMS: " + "; ".join(miss))
        try:
            sents = _clean(sents + llm.complete_json(FILL_SYSTEM, user, Sentences).sentences)
            stats.update(sentences=len(sents), missing_after=len(missing_items(text, sents)))
        except Exception as e:                       # bổ sung hỏng → giữ câu đã có
            stats["fill_error"] = repr(e)[:200]
    return sents, stats
