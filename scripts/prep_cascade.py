"""Dựng lại hai bộ kiểu CascadeKG (Ni et al., KDD 2025 ws / Findings ACL 2026) từ dữ liệu thô + luật đã công bố.

Repo CascadeKG (github.com/Arstanley/cascadekg) chỉ phát hành script sinh và bảng luật, không phát hành tệp
ICEWS14-Event / YAGO-Event; đoạn văn mô tả sự kiện của họ do LLM sinh và cũng không có. Script này sinh lại
theo đúng luật (data/icews14/rule_based_data.py: RULES "add"; data/YAGO3-10/inference_rules.json), nên số mẫu
KHÁC bài báo (test 107 / 2.034); mỗi mẫu là một bài toán cập nhật KG: KG-trước, thao tác tường minh (trigger),
KG-sau = KG-trước + hệ quả gián tiếp theo luật.

Đầu ra: <out>/icews14_event.jsonl, <out>/yago_event.jsonl (định dạng kgu.data.jsonl: idx, event, season, text,
mentioned, before, after, extra) và <out>/prep_stats.json.

--global: giao thức "KG cho trước" (định dạng header base + chênh lệch của kgu.data.jsonl, cumulative=false):
  ICEWS14-Event: KG nền = TOÀN BỘ triple duy nhất của icews_2014_train.txt (42.690 cạnh, 6.869 thực thể, 230 quan hệ);
    trigger lấy từ icews_2014_test.txt; KG-trước = KG nền − trigger; after = KG-trước + trigger + một vòng luật trên
    KG nền (không cap). Tập quan hệ đóng = 230 quan hệ của KG nền (quan hệ luật ICEWS đều có trong đó).
  YAGO-Event: KG nền = TOÀN BỘ YAGO3-10 train.txt (1.079.040 cạnh, 123.143 thực thể, 37 quan hệ); cùng cách chọn
    mẫu/seed như bản lát cắt nên 400 bài trùng với bản lát cắt, chỉ KG-trước đổi thành cả đồ thị. Tập quan hệ đóng =
    37 quan hệ của KG nền → các quan hệ do luật đặt ra (livesIn, hasGrandchild...) KHÔNG nằm trong tập cho phép:
    vế sinh của YAGO-Event không thể đạt được theo giao thức này (cố ý, để đo), chỉ vế cắt có ý nghĩa.
  Tệp ra: <out>/icews14_event_global.jsonl, <out>/yago_event_global.jsonl (tệp YAGO ~45 MB, không commit; kernel sinh lại).

ICEWS14-Event: trigger (A, r, B, ngày d) lấy từ icews_2014_test.txt với r thuộc vế "explicit" của luật ADD;
  KG-trước = mọi cạnh có ngày <= d chạm A hoặc B (trừ chính trigger), giữ tối đa --icews-cap cạnh gần ngày d nhất
  mỗi đầu; luật: (A, r_trig, B) + (B, r_cond, C) [B–C bất kỳ chiều] → (A, r_new, C). `after` = một vòng luật
  (vùng 1-hop mà pipeline khoanh được); `extra.after_full` = 3 vòng (cạnh mới làm trigger tiếp, D phải nằm trong
  KG-trước) để đo phần nhiều bước. Chỉ giữ trigger có >= 1 cạnh gián tiếp. Bài báo cũng chỉ dùng cạnh THÊM.
YAGO-Event: từ YAGO3-10 train.txt; đường (h1, r1, t1), (t1, r2, t2) khớp luật → cạnh mới (h1, r_new, t2)
  (mode cross: cả hai chiều). Hai loại sự kiện như script gốc: "add" (KG-trước chưa có trigger lẫn cạnh suy ra;
  thao tác tường minh ADD trigger) và "delete" (KG-trước có cả; thao tác INVALIDATE trigger, cạnh suy ra phải mất).
  Vai trò của trigger: role=hop1 (trigger là cạnh đầu đường, h1–t1) hoặc role=hop2 (trigger là cạnh thứ hai
  t1–t2, script gốc cũng sinh) — khoanh vùng 1-hop quanh (a, b) chỉ với tới hệ quả (a, ·) nên hop2 dự kiến
  ngoài vùng; để nguyên để đo. KG-trước = 1-hop của hai đầu trigger, ưu tiên cạnh chạm entity trong cạnh suy ra,
  tối đa --yago-cap cạnh.
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ICEWS_URL = "https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews14/icews_2014_{split}.txt"
YAGO_URL = "https://raw.githubusercontent.com/TimDettmers/ConvE/master/YAGO3-10.tar.gz"

# Luật ICEWS14-Event (RULES op=add của CascadeKG; tên quan hệ đúng như trong icews_2014_*.txt)
ICEWS_RULES = [
    ("Sign formal agreement", "Engage in diplomatic cooperation", "Engage in diplomatic cooperation"),
    ("Grant diplomatic recognition", "Engage in diplomatic cooperation", "Engage in diplomatic cooperation"),
    ("Provide economic aid", "Cooperate economically", "Provide economic aid"),
    ("Provide military aid", "Cooperate militarily", "Provide military aid"),
    ("Mobilize or increase armed forces", "Cooperate militarily", "Increase military alert status"),
    ("Declare truce, ceasefire", "Engage in diplomatic cooperation", "Express intent to de-escalate military engagement"),
    ("Ease economic sanctions, boycott, embargo", "Cooperate economically", "Cooperate economically"),
    ("Sign formal agreement", "Cooperate militarily", "Cooperate militarily"),
    ("Sign formal agreement", "Share intelligence or information", "Share intelligence or information"),
    ("Provide military aid", "Express intent to cooperate on intelligence", "Express intent to cooperate on intelligence"),
    ("Provide economic aid", "Provide humanitarian aid", "Provide humanitarian aid"),
    ("Occupy territory", "Engage in diplomatic cooperation", "Complain officially"),
    ("Impose embargo, boycott, or sanctions", "Engage in diplomatic cooperation", "Threaten with sanctions, boycott, embargo"),
    ("Grant diplomatic recognition", "Engage in diplomatic cooperation", "Grant diplomatic recognition"),
    ("Provide humanitarian aid", "Engage in diplomatic cooperation", "Engage in material cooperation"),
    ("Ease economic sanctions, boycott, embargo", "Engage in diplomatic cooperation", "Make optimistic comment"),
]
# Luật YAGO-Event (inference_rules.json của CascadeKG, bỏ trùng)
YAGO_RULES = [
    (("graduatedFrom", "isLocatedIn"), "wasEducatedIn", "forward"),
    (("hasChild", "hasChild"), "hasGrandchild", "forward"),
    (("isLeaderOf", "isLocatedIn"), "livesIn", "forward"),
    (("owns", "isLocatedIn"), "hasBusinessIn", "forward"),
    (("actedIn", "directed"), "collaboratedWith", "cross"),
    (("actedIn", "wroteMusicFor"), "sharedCreativeWork", "cross"),
    (("isAffiliatedTo", "hasNeighbor"), "collaboratesWith", "forward"),
    (("created", "exports"), "tradesWith", "forward"),
    (("hasAcademicAdvisor", "graduatedFrom"), "academicLineageFrom", "forward"),
    (("playsFor", "isLocatedIn"), "livesIn", "forward"),
    (("isCitizenOf", "hasCapital"), "residesInCapital", "forward"),
    (("hasAcademicAdvisor", "hasAcademicAdvisor"), "academicAncestor", "forward"),
    (("directed", "edited"), "sharedCreativeWork", "cross"),
    (("graduatedFrom", "hasNeighbor"), "hasAcademicConnectionTo", "forward"),
    (("created", "isLocatedIn"), "hasOfficeIn", "forward"),
    (("graduatedFrom", "hasCapital"), "studiedNearCapital", "forward"),
    (("hasChild", "graduatedFrom"), "childGraduatedFrom", "forward"),
    (("actedIn", "edited"), "sharedCreativeWork", "cross"),
    (("directed", "edited"), "coDirectedOrEdited", "cross"),
]

Triple = tuple[str, str, str]


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("tải", url, flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------- ICEWS14-Event
def load_icews(raw: Path) -> list[tuple[str, str, str, date]]:
    rows = []
    for split in ("train", "valid", "test"):
        p = fetch(ICEWS_URL.format(split=split), raw / f"icews_2014_{split}.txt")
        for line in p.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            s, r, o, d = line.rstrip("\n").split("\t")
            rows.append((s, r, o, date.fromisoformat(d)))
    return rows


def icews_rules_once(edges: set[Triple], triggers: list[Triple]) -> set[Triple]:
    """Một vòng: với mỗi trigger (A, r, B) và cạnh B–C (bất kỳ chiều) trong `edges` → (A, r_new, C)."""
    nb: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for h, r, t in edges:
        nb[h].add((r, t))
        nb[t].add((r, h))
    out: set[Triple] = set()
    for a, r_trig, b in triggers:
        for r_exp, r_cond, r_new in ICEWS_RULES:
            if r_exp != r_trig:
                continue
            for r_bc, c in nb[b]:
                if r_bc == r_cond and c != a and c != b:
                    out.add((a, r_new, c))
    return out


def load_icews_split(raw: Path, split: str) -> list[tuple[str, str, str, date]]:
    p = fetch(ICEWS_URL.format(split=split), raw / f"icews_2014_{split}.txt")
    rows = []
    for line in p.read_text(encoding="utf8").splitlines():
        if line.strip():
            s, r, o, d = line.rstrip("\n").split("\t")
            rows.append((s, r, o, date.fromisoformat(d)))
    return rows


def build_icews(raw: Path, n: int, cap: int, rng: random.Random, global_kg: bool = False) -> tuple[list[dict], dict]:
    rows = load_icews(raw)
    by_ent: dict[str, list[tuple[str, str, str, date]]] = defaultdict(list)
    for row in rows:
        by_ent[row[0]].append(row)
        by_ent[row[2]].append(row)
    explicit_rels = {r for r, _, _ in ICEWS_RULES}
    if global_kg:                                       # KG nền = toàn bộ train; trigger chỉ lấy từ test
        base: set[Triple] = {(h, rr, t) for h, rr, t, _ in load_icews_split(raw, "train")}
        test = [row for row in load_icews_split(raw, "test") if row[1] in explicit_rels]
    else:
        base = set()
        test = [row for row in rows if row[1] in explicit_rels]
    rng.shuffle(test)
    out, stats = [], {"candidates": len(test), "no_implicit": 0, "n_before": [], "n_add1": [], "n_add_full": []}
    if global_kg:
        stats.update({"base_triples": len(base), "base_entities": len({e for h, _, t in base for e in (h, t)}),
                      "relations": len({rr for _, rr, _ in base}), "trigger_in_base": 0})
    seen: set[tuple] = set()
    for a, r, b, d in test:
        if (a, r, b, d) in seen:
            continue
        seen.add((a, r, b, d))
        trigger = (a, r, b)
        if global_kg:
            before = base - {trigger}
            stats["trigger_in_base"] += trigger in base
        else:
            before = set()
            for e in (a, b):
                hist = sorted((row for row in by_ent[e] if row[3] <= d and (row[0], row[1], row[2]) != (a, r, b)),
                              key=lambda row: row[3], reverse=True)[:cap]
                before |= {(h, rr, t) for h, rr, t, _ in hist}
        add1 = icews_rules_once(before | {trigger}, [trigger]) - before
        if not add1:
            stats["no_implicit"] += 1
            continue
        # 3 vòng: cạnh mới (A, r_new, C) lại làm trigger với pivot C, D phải nằm trong KG-trước
        cur, full, new = before | {trigger} | add1, set(add1), set(add1)
        for _ in range(2):
            new = icews_rules_once(cur, sorted(new)) - cur
            if not new:
                break
            cur |= new
            full |= new
        ex = {
            "idx": len(out), "event": r, "season": d.isoformat(),
            "text": f"On {d.isoformat()}, {a} took the following action toward {b}: {r.lower()}.",
            "mentioned": [a, b],
            "extra": {"trigger": [a, r, b], "op": "ADD", "n_add1": len(add1), "n_add_full": len(full)},
        }
        if global_kg:                                   # chênh lệch so với KG nền / KG-trước (after_full: chỉ phần thêm)
            ex.update({"before_add": [], "before_del": sorted(base - before),
                       "after_add": sorted({trigger} | add1), "after_del": [],
                       "extra": {**ex["extra"], "after_full_add": sorted({trigger} | full)}})
        else:
            ex.update({"before": sorted(before), "after": sorted(before | {trigger} | add1),
                       "extra": {**ex["extra"], "after_full": sorted(before | {trigger} | full)}})
        stats["n_before"].append(len(before))
        stats["n_add1"].append(len(add1))
        stats["n_add_full"].append(len(full))
        out.append(ex)
        if len(out) >= n:
            break
    if global_kg:
        stats["_base"] = sorted(base)
    return out, stats


# ---------------------------------------------------------------------------------------------- YAGO-Event
def load_yago(raw: Path) -> list[Triple]:
    tgz = fetch(YAGO_URL, raw / "YAGO3-10.tar.gz")
    train = raw / "YAGO3-10" / "train.txt"
    if not train.exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(raw / "YAGO3-10")
        if not train.exists():                         # tar có thể không có thư mục con
            for p in (raw / "YAGO3-10").rglob("train.txt"):
                train = p
                break
    triples = []
    for line in train.read_text(encoding="utf8").splitlines():
        if line.strip():
            h, r, t = line.rstrip("\n").split("\t")
            triples.append((h, r, t))
    return triples


def build_yago(raw: Path, n: int, cap: int, rng: random.Random, global_kg: bool = False) -> tuple[list[dict], dict]:
    triples = load_yago(raw)
    base: set[Triple] = set(triples) if global_kg else set()
    out_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    in_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for h, r, t in triples:
        out_adj[h].append((r, t))
        in_adj[t].append((r, h))
    rules_by_path: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for path, new_rel, mode in YAGO_RULES:
        rules_by_path[path].append((new_rel, mode))
    r1s = {p[0] for p in rules_by_path}
    cand = [tr for tr in triples if tr[1] in r1s]
    rng.shuffle(cand)
    kinds = ["add", "delete"]
    quota = {("hop1", "add"): n * 3 // 8, ("hop1", "delete"): n * 3 // 8, ("hop2", "add"): n - 2 * (n * 3 // 8)}
    got: dict[tuple[str, str], int] = defaultdict(int)
    out, stats = [], {"candidates": len(cand), "n_before": [], "n_implicit": [], "quota": {f"{a}/{b}": v for (a, b), v in quota.items()}}

    def one_hop(e: str) -> set[Triple]:
        return {(e, r, t) for r, t in out_adj[e]} | {(h, r, e) for r, h in in_adj[e]}

    for h1, r1, t1 in cand:
        if sum(got.values()) >= n:
            break
        inferred: set[Triple] = set()
        second: list[Triple] = []
        for r2, t2 in out_adj[t1]:
            for new_rel, mode in rules_by_path.get((r1, r2), ()):
                if t2 == h1:
                    continue
                inferred.add((h1, new_rel, t2))
                if mode == "cross":
                    inferred.add((t2, new_rel, h1))
                second.append((t1, r2, t2))
        if not inferred:
            continue
        role = "hop1"
        trigger = (h1, r1, t1)
        if got[("hop1", "add")] >= quota[("hop1", "add")] and got[("hop1", "delete")] >= quota[("hop1", "delete")]:
            role, trigger = "hop2", second[0]                      # trigger là cạnh thứ hai của đường
            # chỉ giữ cạnh suy ra đi qua đúng cạnh thứ hai này
            inferred = {tr for tr in inferred for new_rel, mode in rules_by_path.get((r1, trigger[1]), ())
                        if tr[1] == new_rel and trigger[2] in (tr[0], tr[2])}
            if not inferred:
                continue
        kind = "add" if role == "hop2" else min(kinds, key=lambda k: got[(role, k)] / max(1, quota[(role, k)]))
        if got[(role, kind)] >= quota[(role, kind)]:
            continue
        a, b = trigger[0], trigger[2]
        involved = {e for tr in inferred for e in (tr[0], tr[2])} | {h1, t1, a, b}
        hop = (one_hop(a) | one_hop(b)) - {trigger} - inferred
        prio = sorted(tr for tr in hop if tr[0] in involved and tr[2] in involved)
        rest = sorted(hop - set(prio))
        rng.shuffle(rest)                                       # giữ nguyên thứ tự ngẫu nhiên → cùng 400 bài khi --global
        before = set(prio[:cap]) | set(rest[: max(0, cap - len(prio))])
        if role == "hop2" and (h1, r1, t1) not in before:
            before.add((h1, r1, t1))                            # đường cần cạnh đầu để suy ra
        if global_kg:
            before = base - {trigger} - inferred                # KG nền trọn vẹn (trigger ∈ train.txt được rút ra khi "add")
        if kind == "add":
            after = before | {trigger} | inferred
            op = "ADD"
        else:
            before = before | {trigger} | inferred
            after = before - {trigger} - inferred
            op = "INVALIDATE"
        verb = {"add": "A new fact is recorded", "delete": "A fact is retracted"}[kind]
        text = f"{verb}: {trigger[0].replace('_', ' ')} — {trigger[1]} — {trigger[2].replace('_', ' ')}."
        row = {
            "idx": len(out), "event": f"{role}/{kind}", "season": "", "text": text, "mentioned": [a, b],
            "extra": {"trigger": list(trigger), "op": op, "role": role, "path": [h1, r1, t1],
                      "n_implicit": len(inferred)},
        }
        if global_kg:
            row.update({"before_add": sorted(before - base), "before_del": sorted(base - before),
                        "after_add": sorted(after - before), "after_del": sorted(before - after)})
        else:
            row.update({"before": sorted(before), "after": sorted(after)})
        out.append(row)
        got[(role, kind)] += 1
        stats["n_before"].append(len(before))
        stats["n_implicit"].append(len(inferred))
    stats["got"] = {f"{a}/{b}": v for (a, b), v in got.items()}
    if global_kg:
        stats.update({"base_triples": len(base), "base_entities": len({e for h, _, t in base for e in (h, t)}),
                      "relations": len({rr for _, rr, _ in base}), "_base": sorted(base)})
    return out, stats


def _agg(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            out[k] = {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v)}
        else:
            out[k] = v
    return out


def dump_global(path: Path, rows: list[dict], stats: dict) -> None:
    base = stats.pop("_base")
    header = {"base": base, "cumulative": False, "relations": sorted({r for _, r, _ in base})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/cascade"))
    ap.add_argument("--out", type=Path, default=Path("data/cascade"))
    ap.add_argument("--n-icews", type=int, default=300)
    ap.add_argument("--n-yago", type=int, default=400)
    ap.add_argument("--icews-cap", type=int, default=120, help="cạnh gần nhất mỗi đầu trigger (ICEWS)")
    ap.add_argument("--yago-cap", type=int, default=80, help="cạnh 1-hop tối đa trong KG-trước (YAGO)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["icews", "yago"], default=None)
    ap.add_argument("--global", dest="global_kg", action="store_true", help="giao thức KG cho trước (xem docstring)")
    args = ap.parse_args()
    suffix = "_global" if args.global_kg else ""
    stats_all = {}
    if args.only in (None, "icews"):
        rows, st = build_icews(args.raw, args.n_icews, args.icews_cap, random.Random(args.seed), args.global_kg)
        if args.global_kg:
            dump_global(args.out / f"icews14_event{suffix}.jsonl", rows, st)
        else:
            dump(args.out / "icews14_event.jsonl", rows)
        stats_all["icews14_event" + suffix] = _agg(st)
        print("ICEWS14-Event:", json.dumps(stats_all["icews14_event" + suffix], ensure_ascii=False), flush=True)
    if args.only in (None, "yago"):
        rows, st = build_yago(args.raw, args.n_yago, args.yago_cap, random.Random(args.seed), args.global_kg)
        if args.global_kg:
            dump_global(args.out / f"yago_event{suffix}.jsonl", rows, st)
        else:
            dump(args.out / "yago_event.jsonl", rows)
        stats_all["yago_event" + suffix] = _agg(st)
        print("YAGO-Event:", json.dumps(stats_all["yago_event" + suffix], ensure_ascii=False), flush=True)
    sp = args.out / f"prep_stats{suffix}.json"
    old = json.loads(sp.read_text(encoding="utf8")) if sp.exists() else {}
    old.update(stats_all)
    sp.write_text(json.dumps(old, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
