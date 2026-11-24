"""Sinh bộ dữ liệu tổng hợp KGU-Synth để đo độ phủ và độ chính xác khi tìm thay đổi gián tiếp theo độ sâu.

Không lấy dữ liệu từ nguồn nào: một KG nền hư cấu gồm ba miền (bóng đá, doanh nghiệp, đại học) được dựng từ seed;
mỗi mẫu là MỘT sự kiện đổi vài fact gốc (thay đổi trực tiếp, bản tin nói thẳng), còn thay đổi gián tiếp là phần chênh
lệch của các fact SUY RA theo luật của thế giới (cùng CLB thì là đồng đội, công ty đặt trụ sở ở đâu thì nhân viên làm
việc ở đó...). Mọi quan hệ suy ra đều đã có sẵn rất nhiều trong KG nền, nên không mẫu nào tạo quan hệ mới; khoảng 3%
số mẫu có một thực thể mới (người vừa ký hợp đồng).

Nhãn độ sâu của một thay đổi gián tiếp = số tầng luật kể từ fact gốc bị đổi (1, 2 hoặc 3), ví dụ
    member_of(e, phòng)  ->  works_for(e, công ty) [1]  ->  based_in(e, thành phố) [2]  ->  resident_of(e, nước) [3].

Đầu ra dùng định dạng "KG cho trước" của kgu.data.jsonl (dòng đầu là KG nền, mỗi mẫu chỉ ghi chênh lệch).

Ba lệnh:
    python scripts/gen_synth.py briefs --out-dir DIR [--chunks 10]     # đề bài cho người viết bản tin
    python scripts/gen_synth.py check  --texts FILE_OR_DIR              # kiểm bản tin đã viết
    python scripts/gen_synth.py build  --texts FILE_OR_DIR --out data/synth/kgu_synth.jsonl
                                       [--placeholder]                  # ghép bản tin + nhãn vàng thành bộ dữ liệu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

Triple = tuple[str, str, str]

SEED = 42


# ---------------------------------------------------------------------------------------------------------------
# Tên hư cấu
# ---------------------------------------------------------------------------------------------------------------

_ROOT_A = ["Vel", "Zor", "Kyr", "Tam", "Bel", "Ond", "Syr", "Quel", "Dra", "Mor", "Nev", "Ish", "Ar", "Lum", "Fen",
           "Oro", "Ty", "Hes", "Cal", "Ev", "Jor", "Pra", "Sel", "Ulm", "Wyn", "Xan", "Yst", "Bry", "Cas", "Dov"]
_ROOT_B = ["aria", "enna", "oth", "umar", "essa", "ador", "ivar", "orin", "ymbra", "anthe", "elun", "osca", "adri",
           "ithe", "uvia", "arak", "omer", "elle", "ovia", "yra"]
_CLUB_SUF = ["FC", "United", "Rovers", "Athletic", "City", "Wanderers"]
_STADIUM_SUF = ["Park", "Arena", "Stadium", "Ground"]
_UNIV_SUF = ["University", "Institute of Technology", "Polytechnic", "State University"]
_COMPANY_SUF = ["Dynamics", "Systems", "Logistics", "Foods", "Energy", "Pharma", "Robotics", "Textiles", "Mobility",
                "Analytics", "Motors", "Insurance"]
_LEAGUE_SUF = ["Premier League", "First Division", "Championship", "Super League"]
_FED_SUF = ["Football Federation", "Football Union", "Football Association"]
_CODENAME = ["Atlas", "Borealis", "Cobalt", "Delta", "Ember", "Falcon", "Granite", "Harbor", "Indigo", "Juniper",
             "Kestrel", "Lantern", "Meridian", "Nimbus", "Onyx", "Pioneer", "Quartz", "Raven", "Summit", "Tundra",
             "Umber", "Vector", "Willow", "Zenith", "Aspen", "Beacon", "Cedar", "Drift", "Echo", "Flint", "Garnet",
             "Helix", "Ion", "Jade", "Keystone", "Lumen", "Mosaic", "Nova", "Orchid", "Prism"]
_DEPT_FUNC = ["Finance", "Procurement", "Logistics", "Marketing", "Platform", "Research", "Legal", "Sales", "Support",
              "Design", "Quality", "Security", "Data", "Operations", "Payroll", "Compliance"]
_DEPT_UNIT = ["Team", "Division", "Unit", "Group", "Department"]
_LAB_FIELD = ["Photonics", "Genomics", "Robotics", "Climate", "Neuroscience", "Materials", "Linguistics", "Quantum",
              "Hydrology", "Cryptography", "Astrophysics", "Immunology", "Economics", "Seismology", "Polymer", "Vision"]
_LAB_UNIT = ["Lab", "Laboratory", "Research Group"]
_FIRST = ["Tomas", "Elena", "Kofi", "Mirela", "Jonas", "Aiko", "Rafael", "Ingrid", "Dmitri", "Leila", "Mateo", "Sanna",
          "Olu", "Petra", "Hamid", "Yuki", "Luca", "Amara", "Viktor", "Noor", "Emil", "Carmen", "Tariq", "Freya",
          "Andrei", "Priya", "Marco", "Zofia", "Idris", "Hana", "Stefan", "Maren", "Diego", "Anika", "Pavel", "Salma",
          "Henrik", "Ines", "Bashir", "Mei", "Goran", "Lucia", "Kwame", "Tova", "Rui", "Esther", "Milan", "Farah",
          "Oskar", "Nadia", "Javier", "Liv", "Samir", "Chiara", "Anton", "Rosa", "Yusuf", "Greta", "Felipe", "Daria",
          "Nils", "Imani", "Sergio", "Mina", "Bram", "Alba", "Karim", "Elif", "Teodor", "Malia", "Ravi", "Signe",
          "Hugo", "Thandi", "Ivo", "Clara", "Arman", "Sofia", "Lars", "Bianca", "Omar", "Katya", "Dario", "Runa",
          "Aziz", "Marta", "Finn", "Laila", "Tiago", "Vera"]
_LAST = ["Varga", "Okafor", "Lindqvist", "Moreno", "Petrov", "Nakamura", "Haddad", "Kowalski", "Mensah", "Bergstrom",
         "Castillo", "Novak", "Rahimi", "Sorensen", "Adeyemi", "Ferrari", "Dragan", "Tanaka", "Yilmaz", "Brandt",
         "Santos", "Horvat", "Karimi", "Nystrom", "Okoye", "Ricci", "Popescu", "Ishikawa", "Mansour", "Wojcik",
         "Boateng", "Lundgren", "Herrera", "Kovac", "Najafi", "Dahl", "Balogun", "Conti", "Ionescu", "Mori", "Saleh",
         "Zielinski", "Asante", "Holm", "Vargas", "Jovanovic", "Hosseini", "Nielsen", "Eze", "Bianchi", "Stoica",
         "Sato", "Farouk", "Kaminski", "Owusu", "Ekstrom", "Rojas", "Babic", "Rezaei", "Jensen", "Nwosu", "Greco",
         "Marin", "Kimura", "Khalil", "Nowak", "Darko", "Sandberg", "Ortega", "Petrovic", "Azizi", "Olsen", "Obi",
         "Rinaldi", "Lupu", "Hayashi", "Nasser", "Mazur", "Appiah", "Forsberg", "Navarro", "Simic", "Tehrani", "Lund",
         "Uche", "Gallo", "Dumitru", "Inoue", "Aziz", "Sikora", "Ansah", "Hedlund", "Paredes", "Vukovic", "Sharifi",
         "Krogh", "Chukwu", "Marchetti", "Radu", "Ueda", "Hamdan", "Pawlak", "Addo", "Oberg", "Salazar", "Ilic",
         "Ghorbani", "Thorsen", "Okeke", "Lombardi"]

_PUNCT = ".,;:!?\"'()[]{}“”‘’—–-…"


def _norm_tokens(text: str) -> list[str]:
    """Tách từ để dò tên thực thể: bỏ dấu câu hai đầu và đuôi sở hữu ('s)."""
    out = []
    for tok in text.split():
        tok = tok.strip(_PUNCT)
        for suf in ("'s", "’s"):
            if tok.endswith(suf):
                tok = tok[: -len(suf)]
        tok = tok.strip(_PUNCT)
        if tok:
            out.append(tok)
    return out


class Names:
    """Cấp tên duy nhất; không tên nào là dãy từ liên tiếp nằm trong tên khác (để dò rò rỉ thực thể trong bản tin)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        roots = [a + b for a in _ROOT_A for b in _ROOT_B]
        rng.shuffle(roots)
        self._roots = roots
        self._taken: set[str] = set()

    def root(self) -> str:
        return self._roots.pop()

    def _unique(self, make) -> str:
        for _ in range(10_000):
            name = make()
            if name not in self._taken:
                self._taken.add(name)
                return name
        raise RuntimeError("hết tên")

    def place(self, suffixes: list[str] | None = None) -> str:
        return self._unique(lambda: self.root() + (" " + self.rng.choice(suffixes) if suffixes else ""))

    def person(self) -> str:
        return self._unique(lambda: f"{self.rng.choice(_FIRST)} {self.rng.choice(_LAST)}")

    def dept(self) -> str:
        return self._unique(lambda: f"{self.rng.choice(_CODENAME)} {self.rng.choice(_DEPT_FUNC)} {self.rng.choice(_DEPT_UNIT)}")

    def lab(self) -> str:
        return self._unique(lambda: f"{self.rng.choice(_CODENAME)} {self.rng.choice(_LAB_FIELD)} {self.rng.choice(_LAB_UNIT)}")


def nested_names(names: list[str]) -> list[tuple[str, str]]:
    """Các cặp (tên ngắn, tên dài) mà tên ngắn là dãy từ liên tiếp trong tên dài. Bộ dữ liệu hợp lệ thì rỗng."""
    token_names = {tuple(n.split()): n for n in names}
    bad = []
    for toks, n in token_names.items():
        for size in range(1, len(toks)):
            for i in range(len(toks) - size + 1):
                sub = toks[i:i + size]
                if sub in token_names:
                    bad.append((token_names[sub], n))
    return bad


# ---------------------------------------------------------------------------------------------------------------
# Thế giới: fact gốc (đổi được bằng sự kiện) + fact độc lập (không bao giờ đổi) -> bao đóng theo luật
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    city_country: dict[str, str]
    # bóng đá
    league_fed: dict[str, str]
    club_league: dict[str, str]
    club_coach: dict[str, str | None]
    player_club: dict[str, str | None]
    # doanh nghiệp
    company_city: dict[str, str]
    dept_company: dict[str, str]
    dept_head: dict[str, str | None]
    emp_dept: dict[str, str | None]
    # đại học
    univ_city: dict[str, str]
    lab_univ: dict[str, str]
    lab_director: dict[str, str | None]
    res_lab: dict[str, str | None]
    static: frozenset[Triple] = field(default_factory=frozenset)       # fact độc lập: quê quán, bạn bè, tài trợ...
    etype: dict[str, str] = field(default_factory=dict)                 # thực thể -> loại (để viết đề bài)

    def mutate(self, muts: list[tuple[str, str, str | None]]) -> "World":
        """Bản sao với vài fact gốc bị đổi: (tên bảng, khóa, giá trị mới)."""
        changed: dict[str, dict] = {}
        for table, key, value in muts:
            changed.setdefault(table, dict(getattr(self, table)))[key] = value
        return replace(self, **changed)
# Quan hệ suy ra và tầng luật của chúng; mọi quan hệ còn lại là fact gốc hoặc fact độc lập.
DERIVED = ("teammate_of", "coached_by", "plays_in", "registered_with",
           "colleague_of", "reports_to", "works_for",
           "labmate_of", "supervised_by", "affiliated_with",
           "based_in", "resident_of")


def closure(w: World) -> dict[Triple, tuple[Triple, ...]]:
    """Mọi triple đang đúng trong thế giới `w` -> các tiền đề của nó (rỗng nếu là fact gốc hay fact độc lập)."""
    out: dict[Triple, tuple[Triple, ...]] = {tr: () for tr in w.static}
    for city, country in w.city_country.items():
        out[(city, "located_in", country)] = ()

    def place_chain(x: str, org_fact: Triple, org: str, hq_fact: Triple, city: str) -> None:
        """x gắn với tổ chức `org` (fact org_fact, suy ra) -> based_in thành phố của org -> resident_of nước."""
        based = (x, "based_in", city)
        out[based] = (org_fact, hq_fact)
        country = w.city_country[city]
        out[(x, "resident_of", country)] = (based, (city, "located_in", country))

    # ---- bóng đá: plays_for -> teammate_of, coached_by, plays_in -> registered_with
    for league, fed in w.league_fed.items():
        out[(league, "governed_by", fed)] = ()
    squads: dict[str, list[str]] = defaultdict(list)
    for p, club in w.player_club.items():
        if club:
            squads[club].append(p)
    for club, league in w.club_league.items():
        out[(club, "competes_in", league)] = ()
        coach = w.club_coach.get(club)
        if coach:
            out[(coach, "coaches", club)] = ()
        fed = w.league_fed[league]
        squad = squads.get(club, ())
        for p in squad:
            pf = (p, "plays_for", club)
            out[pf] = ()
            for q in squad:
                if q != p:
                    out[(p, "teammate_of", q)] = (pf, (q, "plays_for", club))
            if coach:
                out[(p, "coached_by", coach)] = (pf, (coach, "coaches", club))
            pin = (p, "plays_in", league)
            out[pin] = (pf, (club, "competes_in", league))
            out[(p, "registered_with", fed)] = (pin, (league, "governed_by", fed))

    # ---- doanh nghiệp: member_of / heads -> colleague_of, reports_to, works_for -> based_in -> resident_of
    for company, city in w.company_city.items():
        out[(company, "headquartered_in", city)] = ()
    staff: dict[str, list[str]] = defaultdict(list)
    for e, dept in w.emp_dept.items():
        if dept:
            staff[dept].append(e)
    for dept, company in w.dept_company.items():
        part = (dept, "part_of", company)
        out[part] = ()
        city = w.company_city[company]
        hq = (company, "headquartered_in", city)
        head = w.dept_head.get(dept)
        if head:
            hd = (head, "heads", dept)
            out[hd] = ()
            wf = (head, "works_for", company)
            out[wf] = (hd, part)
            place_chain(head, wf, company, hq, city)
        members = staff.get(dept, ())
        for e in members:
            mo = (e, "member_of", dept)
            out[mo] = ()
            for e2 in members:
                if e2 != e:
                    out[(e, "colleague_of", e2)] = (mo, (e2, "member_of", dept))
            if head:
                out[(e, "reports_to", head)] = (mo, (head, "heads", dept))
            wf = (e, "works_for", company)
            out[wf] = (mo, part)
            place_chain(e, wf, company, hq, city)

    # ---- đại học: works_in / directs -> labmate_of, supervised_by, affiliated_with -> based_in -> resident_of
    for univ, city in w.univ_city.items():
        out[(univ, "located_in", city)] = ()
    groups: dict[str, list[str]] = defaultdict(list)
    for r, lab in w.res_lab.items():
        if lab:
            groups[lab].append(r)
    for lab, univ in w.lab_univ.items():
        bel = (lab, "belongs_to", univ)
        out[bel] = ()
        city = w.univ_city[univ]
        loc = (univ, "located_in", city)
        director = w.lab_director.get(lab)
        if director:
            dr = (director, "directs", lab)
            out[dr] = ()
            af = (director, "affiliated_with", univ)
            out[af] = (dr, bel)
            place_chain(director, af, univ, loc, city)
        members = groups.get(lab, ())
        for r in members:
            wi = (r, "works_in", lab)
            out[wi] = ()
            for r2 in members:
                if r2 != r:
                    out[(r, "labmate_of", r2)] = (wi, (r2, "works_in", lab))
            if director:
                out[(r, "supervised_by", director)] = (wi, (director, "directs", lab))
            af = (r, "affiliated_with", univ)
            out[af] = (wi, bel)
            place_chain(r, af, univ, loc, city)
    return out


def build_world(rng: random.Random) -> tuple[World, Names, dict[str, list[str]]]:
    """KG nền. Trả thêm các nhóm người chưa gắn tổ chức (cầu thủ tự do, người chưa có việc...) để làm sự kiện ký mới."""
    nm = Names(rng)
    etype: dict[str, str] = {}

    def new(kind: str, name: str) -> str:
        etype[name] = kind
        return name
    countries = [new("country", nm.place()) for _ in range(8)]
    cities = [new("city", nm.place()) for _ in range(24)]
    city_country = {c: countries[i // 3] for i, c in enumerate(cities)}
    # bóng đá: 3 liên đoàn, 8 giải, 48 CLB
    feds = [new("football federation", nm.place(_FED_SUF)) for _ in range(3)]
    leagues = [new("league", nm.place(_LEAGUE_SUF)) for _ in range(8)]
    league_fed = {lg: feds[[0, 0, 0, 1, 1, 1, 2, 2][i]] for i, lg in enumerate(leagues)}
    clubs = [new("football club", nm.place(_CLUB_SUF)) for _ in range(48)]
    club_league = {c: leagues[i // 6] for i, c in enumerate(clubs)}
    coachless = set(rng.sample(clubs, 4))
    club_coach = {c: (None if c in coachless else new("football coach", nm.person())) for c in clubs}
    player_club: dict[str, str | None] = {}
    for c in clubs:
        for _ in range(rng.randint(11, 16)):
            player_club[new("football player", nm.person())] = c
    free_players = [new("football player", nm.person()) for _ in range(46)]
    player_club.update({p: None for p in free_players})
    free_coaches = [new("football coach", nm.person()) for _ in range(20)]
    stadiums = {c: new("stadium", nm.place(_STADIUM_SUF)) for c in clubs}
    agents = [new("player agent", nm.person()) for _ in range(25)]
    # doanh nghiệp: 36 công ty (12 thành phố đầu có 2 công ty), mỗi công ty 3–4 phòng ban
    companies = [new("company", nm.place(_COMPANY_SUF)) for _ in range(36)]
    company_city = {co: cities[i % 12] if i < 24 else cities[12 + (i - 24)] for i, co in enumerate(companies)}
    dept_company: dict[str, str] = {}
    for co in companies:
        for _ in range(rng.randint(3, 4)):
            dept_company[new("company department", nm.dept())] = co
    depts = list(dept_company)
    headless = set(rng.sample(depts, 8))
    dept_head = {d: (None if d in headless else new("department head", nm.person())) for d in depts}
    emp_dept: dict[str, str | None] = {}
    for d in depts:
        for _ in range(rng.randint(5, 9)):
            emp_dept[new("employee", nm.person())] = d
    jobless = [new("professional", nm.person()) for _ in range(42)]
    emp_dept.update({e: None for e in jobless})
    free_managers = [new("manager", nm.person()) for _ in range(20)]
    # đại học: 18 trường (6 thành phố đầu có 2 trường), mỗi trường 3 phòng thí nghiệm
    univs = [new("university", nm.place(_UNIV_SUF)) for _ in range(18)]
    univ_city = {u: cities[i % 6] if i < 12 else cities[6 + (i - 12) * 3] for i, u in enumerate(univs)}
    lab_univ = {new("research lab", nm.lab()): u for u in univs for _ in range(3)}
    labs = list(lab_univ)
    directorless = set(rng.sample(labs, 5))
    lab_director = {lb: (None if lb in directorless else new("professor", nm.person())) for lb in labs}
    res_lab: dict[str, str | None] = {}
    for lb in labs:
        for _ in range(rng.randint(4, 8)):
            res_lab[new("researcher", nm.person())] = lb
    free_res = [new("researcher", nm.person()) for _ in range(34)]
    res_lab.update({r: None for r in free_res})
    free_profs = [new("professor", nm.person()) for _ in range(14)]

    # ---- fact độc lập (không phụ thuộc sự kiện nào): bẫy cho bước xét duyệt
    static: set[Triple] = set()
    persons = [e for e, k in etype.items() if k in ("football player", "football coach", "player agent", "employee",
                                                    "department head", "professional", "manager", "researcher", "professor")]
    for p in persons:
        static.add((p, "born_in", rng.choice(cities)))
    group_of = {**{p: c for p, c in player_club.items() if c}, **{e: d for e, d in emp_dept.items() if d},
                **{r: lb for r, lb in res_lab.items() if lb}}
    members: dict[str, list[str]] = defaultdict(list)
    for x, g in group_of.items():
        members[g].append(x)
    for p in persons:                                             # bạn bè: khoảng 40% cùng nhóm, còn lại bất kỳ
        if rng.random() < 0.5:
            mates = [m for m in members.get(group_of.get(p, ""), ()) if m != p]
            q = rng.choice(mates) if mates and rng.random() < 0.4 else rng.choice(persons)
            if q != p:
                static |= {(p, "friend_of", q), (q, "friend_of", p)}
    for p in [*player_club, *free_coaches, *[c for c in club_coach.values() if c]]:
        if rng.random() < 0.6:
            static.add((p, "represented_by", rng.choice(agents)))
    for p in persons:
        k = etype[p]
        prob = {"employee": 0.5, "department head": 0.5, "manager": 0.5, "professional": 0.5,
                "researcher": 0.7, "professor": 0.6}.get(k, 0.1)
        if rng.random() < prob:
            static.add((p, "alumni_of", rng.choice(univs)))
    for c in clubs:
        static.add((c, "home_ground", stadiums[c]))
        static.add((stadiums[c], "located_in", rng.choice(cities)))
        for co in rng.sample(companies, rng.randint(1, 2)):
            static.add((co, "sponsors", c))
        for p in members[c]:
            if rng.random() < 0.6:
                static.add((p, "debuted_at", c if rng.random() < 0.5 else rng.choice(clubs)))
    for lb in labs:
        if rng.random() < 0.4:
            static.add((rng.choice(companies), "funds", lb))
    scholars = [p for p in persons if etype[p] in ("researcher", "professor")]
    for r in scholars:                                            # đồng tác giả: khoảng 35% cùng phòng thí nghiệm
        for _ in range(rng.randint(1, 2)):
            mates = [m for m in members.get(group_of.get(r, ""), ()) if m != r]
            q = rng.choice(mates) if mates and rng.random() < 0.35 else rng.choice(scholars)
            if q != r:
                static |= {(r, "coauthor_of", q), (q, "coauthor_of", r)}
    for _ in range(30):
        a, b = rng.sample(companies, 2)
        static.add((a, "supplier_of", b))
    world = World(city_country, league_fed, club_league, club_coach, player_club, company_city, dept_company,
                  dept_head, emp_dept, univ_city, lab_univ, lab_director, res_lab, frozenset(static), etype)
    pools = {"free_players": free_players, "free_coaches": free_coaches, "jobless": jobless,
             "free_managers": free_managers, "free_res": free_res, "free_profs": free_profs}
    return world, nm, pools


# ---------------------------------------------------------------------------------------------------------------
# Sự kiện


# ---------------------------------------------------------------------------------------------------------------

# (họ sự kiện, biến thể, số mẫu). Tổng 1.000. Cột ghi chú: độ sâu lớn nhất của thay đổi gián tiếp.
PLAN: list[tuple[str, str, int]] = [
    # bóng đá (350)
    ("f_transfer", "same_league", 40),        # 1: đồng đội, huấn luyện viên
    ("f_transfer", "other_league", 50),       # 1: + giải đấu
    ("f_transfer", "other_fed", 55),          # 2: + liên đoàn
    ("f_transfer", "swap", 13),               # hai cầu thủ đổi chỗ giữa hai CLB
    ("f_transfer", "double", 12),             # hai đồng đội cùng sang một CLB (quan hệ đồng đội giữa họ giữ nguyên)
    ("f_sign", "free_agent", 40),             # chỉ có vế sinh
    ("f_sign", "new_entity", 10),             # thực thể mới
    ("f_release", "release", 40),             # chỉ có vế cắt
    ("f_coach", "replace", 40),
    ("f_coach", "sack", 12),
    ("f_coach", "appoint", 8),
    ("f_league", "same_fed", 30),             # cả CLB đổi giải: thay đổi của cả nhóm
    # doanh nghiệp (340)
    ("c_move", "same_company", 30),           # 1; works_for giữ nguyên
    ("c_move", "same_city", 25),              # 1; based_in giữ nguyên
    ("c_move", "other_city", 35),             # 2; resident_of giữ nguyên
    ("c_move", "other_country", 35),          # 3
    ("c_move", "double", 15),
    ("c_hire", "jobless", 35),
    ("c_hire", "new_entity", 10),
    ("c_leave", "leave", 35),
    ("c_head", "replace", 35),
    ("c_head", "leave", 8),
    ("c_head", "appoint", 7),
    ("c_hq", "same_country", 18),             # công ty dời trụ sở: thay đổi của cả nhóm
    ("c_hq", "other_country", 12),
    ("c_dept", "same_city", 10),              # phòng ban chuyển sang công ty khác: thay đổi của cả nhóm
    ("c_dept", "other_city", 15),
    ("c_dept", "other_country", 15),
    # đại học (260)
    ("u_move", "same_univ", 25),
    ("u_move", "same_city", 20),
    ("u_move", "other_city", 35),
    ("u_move", "other_country", 30),
    ("u_move", "double", 10),
    ("u_join", "free", 27),
    ("u_join", "new_entity", 8),
    ("u_leave", "leave", 25),
    ("u_director", "replace", 30),
    ("u_director", "leave", 5),
    ("u_director", "appoint", 5),
    ("u_lab", "same_city", 10),               # phòng thí nghiệm chuyển trường: thay đổi của cả nhóm
    ("u_lab", "other_city", 15),
    ("u_lab", "other_country", 15),
    # không có thay đổi nào (50): tin đồn, đàm phán, phủ nhận
    ("no_change", "football", 22),
    ("no_change", "corporate", 16),
    ("no_change", "university", 12),
]

_RUMOUR = ["is reported to be in talks with", "has been linked with a move to", "turned down an approach from",
           "saw a proposed move collapse to", "is said to be considering an offer from"]
STYLES = ["newswire report", "official press statement", "local newspaper item", "one-line news ticker",
          "newsletter paragraph", "radio bulletin script", "trade-press note", "social-media announcement by the organisation"]
EXTRAS = ["mention the weekday or the month (no year)", "include a short quote from an unnamed spokesperson",
          "give a plausible reason for the change", "say when the change takes effect",
          "mention the length of the new contract or term", "add one neutral background detail about the person or organisation "
          "that names no one else", "keep it bare: only the facts"]
EXTRAS_RUMOUR = ["mention the weekday or the month (no year)", "include a short denial or 'no comment' from an unnamed spokesperson",
                 "say where the report comes from (unnamed sources, local media)", "keep it bare: only the facts"]


class Sampler:
    """Chọn thực thể cho từng sự kiện trên KG nền; mỗi cá nhân chỉ làm nhân vật chính của một mẫu."""

    def __init__(self, world: World, names: Names, pools: dict[str, list[str]], rng: random.Random) -> None:
        self.w, self.nm, self.rng = world, names, rng
        self.pools = {k: list(v) for k, v in pools.items()}
        self.used: set[str] = set()
        self.signatures: set[tuple] = set()
        w = world
        self.squad = self._group(w.player_club)
        self.staff = self._group(w.emp_dept)
        self.team = self._group(w.res_lab)
        self.country_of_company = {co: w.city_country[c] for co, c in w.company_city.items()}
        self.country_of_univ = {u: w.city_country[c] for u, c in w.univ_city.items()}
    @staticmethod
    def _group(table: dict[str, str | None]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x, g in table.items():
            if g:
                out[g].append(x)
        return out

    def _person(self, candidates: list[str]) -> str:
        fresh = [c for c in candidates if c not in self.used]
        if not fresh:
            raise LookupError("hết ứng viên")
        x = self.rng.choice(fresh)
        self.used.add(x)
        return x

    def _pool(self, name: str, reuse: bool = False) -> str:
        """Một người chưa gắn tổ chức. reuse=True: nhóm nhỏ (huấn luyện viên, trưởng phòng, giáo sư tự do) được dùng
        lại giữa các mẫu, vì các mẫu độc lập với nhau; trùng hẳn một sự kiện thì `make` đã loại."""
        return self.rng.choice(self.pools[name]) if reuse else self._person(self.pools[name])

    def _t(self, x: str) -> str:
        return f"{x} ({self.w.etype.get(x, 'newcomer')})"
    # mỗi hàm trả dict(muts, mentioned, facts, new_entities)
    def make(self, family: str, variant: str) -> dict:
        for _ in range(500):
            try:
                ev = getattr(self, family)(variant)
            except LookupError:
                continue
            sig = (family, tuple(ev["muts"]), tuple(ev["mentioned"]))
            if sig in self.signatures:
                continue
            self.signatures.add(sig)
            ev.setdefault("new_entities", [])
            return ev
        raise RuntimeError(f"không dựng được sự kiện {family}/{variant}")

    # ---- bóng đá ---------------------------------------------------------------------------------------------
    def _other_club(self, club: str, variant: str) -> str:
        w = self.w
        lg, fed = w.club_league[club], w.league_fed[w.club_league[club]]
        ok = {"same_league": lambda c: w.club_league[c] == lg,
              "other_league": lambda c: w.club_league[c] != lg and w.league_fed[w.club_league[c]] == fed,
              "other_fed": lambda c: w.league_fed[w.club_league[c]] != fed,
              "any": lambda c: True}[variant]
        return self.rng.choice([c for c in w.club_league if c != club and ok(c)])

    def f_transfer(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "swap":
            a = self.rng.choice(list(w.club_league))
            b = self._other_club(a, "any")
            p1, p2 = self._person(self.squad[a]), self._person(self.squad[b])
            return {"muts": [("player_club", p1, b), ("player_club", p2, a)], "mentioned": [p1, p2, a, b],
                    "facts": [f"swap deal between {t(a)} and {t(b)}", f"{t(p1)} leaves {a} and joins {b}",
                              f"{t(p2)} leaves {b} and joins {a}"]}
        if variant == "double":
            a = self.rng.choice(list(w.club_league))
            b = self._other_club(a, "any")
            p1, p2 = self._person(self.squad[a]), self._person(self.squad[a])
            return {"muts": [("player_club", p1, b), ("player_club", p2, b)], "mentioned": [p1, p2, a, b],
                    "facts": [f"{t(p1)} and {t(p2)} both leave {t(a)}", f"both players join {t(b)}"]}
        p = self._person([x for x, c in w.player_club.items() if c])
        a = w.player_club[p]
        b = self._other_club(a, variant)
        return {"muts": [("player_club", p, b)], "mentioned": [p, a, b],
                "facts": [f"{t(p)} leaves {t(a)}", f"{p} joins {t(b)} on a permanent transfer"]}

    def f_sign(self, variant: str) -> dict:
        b = self.rng.choice(list(self.w.club_league))
        if variant == "new_entity":
            p = self.nm.person()
            return {"muts": [("player_club", p, b)], "mentioned": [p, b], "new_entities": [p],
                    "facts": [f"{p} (a young football player with no previous club) signs a first professional contract with {self._t(b)}"]}
        p = self._pool("free_players")
        return {"muts": [("player_club", p, b)], "mentioned": [p, b],
                "facts": [f"{self._t(p)}, a free agent without a club, signs for {self._t(b)}"]}

    def f_release(self, variant: str) -> dict:
        p = self._person([x for x, c in self.w.player_club.items() if c])
        a = self.w.player_club[p]
        how = self.rng.choice(["retires from football", "has the contract terminated and is now without a club",
                               "is released and leaves as a free agent"])
        return {"muts": [("player_club", p, None)], "mentioned": [p, a],
                "facts": [f"{self._t(p)} {how}; the player no longer plays for {self._t(a)}"]}

    def f_coach(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "appoint":
            club = self.rng.choice([c for c, co in w.club_coach.items() if co is None])
            new = self._pool("free_coaches", reuse=True)
            return {"muts": [("club_coach", club, new)], "mentioned": [new, club],
                    "facts": [f"{t(club)}, which had no head coach, appoints {t(new)} as head coach"]}
        club = self.rng.choice([c for c, co in w.club_coach.items() if co])
        old = w.club_coach[club]
        if variant == "sack":
            return {"muts": [("club_coach", club, None)], "mentioned": [old, club],
                    "facts": [f"{t(club)} dismisses head coach {t(old)}; no successor is named, the post is vacant"]}
        new = self._pool("free_coaches", reuse=True)
        return {"muts": [("club_coach", club, new)], "mentioned": [old, new, club],
                "facts": [f"{t(old)} is no longer head coach of {t(club)}", f"{t(new)} takes over as head coach of {club}"]}

    def f_league(self, variant: str) -> dict:
        w = self.w
        club = self.rng.choice(list(w.club_league))
        old = w.club_league[club]
        new = self.rng.choice([lg for lg, f in w.league_fed.items() if lg != old and f == w.league_fed[old]])
        how = self.rng.choice(["is promoted", "is relegated", "is admitted after a league restructuring"])
        return {"muts": [("club_league", club, new)], "mentioned": [club, old, new],
                "facts": [f"{self._t(club)} {how}: it leaves {self._t(old)}", f"{club} now competes in {self._t(new)}"]}

    # ---- doanh nghiệp ----------------------------------------------------------------------------------------
    def _other_dept(self, dept: str, variant: str) -> str:
        w = self.w
        co = w.dept_company[dept]
        city, country = w.company_city[co], self.country_of_company[co]

        def ok(d: str) -> bool:
            co2 = w.dept_company[d]
            if variant == "same_company":
                return co2 == co
            if variant == "same_city":
                return co2 != co and w.company_city[co2] == city
            if variant == "other_city":
                return w.company_city[co2] != city and self.country_of_company[co2] == country
            if variant == "other_country":
                return self.country_of_company[co2] != country
            return co2 != co
        cands = [d for d in w.dept_company if d != dept and ok(d)]
        if not cands:
            raise LookupError(variant)
        return self.rng.choice(cands)

    def c_move(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "double":
            a = self.rng.choice(list(w.dept_company))
            b = self._other_dept(a, "other")
            e1, e2 = self._person(self.staff[a]), self._person(self.staff[a])
            return {"muts": [("emp_dept", e1, b), ("emp_dept", e2, b)], "mentioned": [e1, e2, a, b],
                    "facts": [f"{t(e1)} and {t(e2)} both leave the {t(a)}", f"both join the {t(b)}"]}
        e = self._person([x for x, d in w.emp_dept.items() if d])
        a = w.emp_dept[e]
        b = self._other_dept(a, variant)
        return {"muts": [("emp_dept", e, b)], "mentioned": [e, a, b],
                "facts": [f"{t(e)} leaves the {t(a)}", f"{e} becomes a member of the {t(b)}"]}

    def c_hire(self, variant: str) -> dict:
        b = self.rng.choice(list(self.w.dept_company))
        if variant == "new_entity":
            e = self.nm.person()
            return {"muts": [("emp_dept", e, b)], "mentioned": [e, b], "new_entities": [e],
                    "facts": [f"{e} (a new graduate starting a first job) joins the {self._t(b)} as a member"]}
        e = self._pool("jobless")
        return {"muts": [("emp_dept", e, b)], "mentioned": [e, b],
                "facts": [f"{self._t(e)}, who had no employer, is hired as a member of the {self._t(b)}"]}

    def c_leave(self, variant: str) -> dict:
        e = self._person([x for x, d in self.w.emp_dept.items() if d])
        a = self.w.emp_dept[e]
        how = self.rng.choice(["resigns", "is laid off", "retires", "leaves at the end of the contract"])
        return {"muts": [("emp_dept", e, None)], "mentioned": [e, a],
                "facts": [f"{self._t(e)} {how} and is no longer a member of the {self._t(a)}; no new employer"]}

    def c_head(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "appoint":
            dept = self.rng.choice([d for d, h in w.dept_head.items() if h is None])
            new = self._pool("free_managers", reuse=True)
            return {"muts": [("dept_head", dept, new)], "mentioned": [new, dept],
                    "facts": [f"the {t(dept)}, which had no head, gets {t(new)} as its new head"]}
        dept = self.rng.choice([d for d, h in w.dept_head.items() if h])
        old = w.dept_head[dept]
        if variant == "leave":
            return {"muts": [("dept_head", dept, None)], "mentioned": [old, dept],
                    "facts": [f"{t(old)} steps down as head of the {t(dept)} and leaves; the post stays vacant"]}
        new = self._pool("free_managers", reuse=True)
        return {"muts": [("dept_head", dept, new)], "mentioned": [old, new, dept],
                "facts": [f"{t(old)} is no longer head of the {t(dept)} and leaves", f"{t(new)} is the new head of the {dept}"]}

    def c_hq(self, variant: str) -> dict:
        w = self.w
        co = self.rng.choice(list(w.company_city))
        old = w.company_city[co]
        same = variant == "same_country"
        new = self.rng.choice([c for c in w.city_country if c != old and (w.city_country[c] == w.city_country[old]) == same])
        return {"muts": [("company_city", co, new)], "mentioned": [co, old, new],
                "facts": [f"{self._t(co)} moves its headquarters out of {self._t(old)}",
                          f"the new headquarters of {co} is in {self._t(new)}; the whole company relocates"]}

    def c_dept(self, variant: str) -> dict:
        w = self.w
        dept = self.rng.choice(list(w.dept_company))
        old = w.dept_company[dept]
        new = w.dept_company[self._other_dept(dept, variant)]
        how = self.rng.choice(["is sold", "is spun off and transferred", "is taken over"])
        return {"muts": [("dept_company", dept, new)], "mentioned": [dept, old, new],
                "facts": [f"the {self._t(dept)} {how}: it is no longer part of {self._t(old)}",
                          f"the {dept} is now part of {self._t(new)}; its head and staff stay with the unit"]}

    # ---- đại học ---------------------------------------------------------------------------------------------
    def _other_lab(self, lab: str, variant: str) -> str:
        w = self.w
        u = w.lab_univ[lab]
        city, country = w.univ_city[u], self.country_of_univ[u]

        def ok(lb: str) -> bool:
            u2 = w.lab_univ[lb]
            if variant == "same_univ":
                return u2 == u
            if variant == "same_city":
                return u2 != u and w.univ_city[u2] == city
            if variant == "other_city":
                return w.univ_city[u2] != city and self.country_of_univ[u2] == country
            if variant == "other_country":
                return self.country_of_univ[u2] != country
            return u2 != u
        cands = [lb for lb in w.lab_univ if lb != lab and ok(lb)]
        if not cands:
            raise LookupError(variant)
        return self.rng.choice(cands)

    def u_move(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "double":
            a = self.rng.choice(list(w.lab_univ))
            b = self._other_lab(a, "other")
            r1, r2 = self._person(self.team[a]), self._person(self.team[a])
            return {"muts": [("res_lab", r1, b), ("res_lab", r2, b)], "mentioned": [r1, r2, a, b],
                    "facts": [f"{t(r1)} and {t(r2)} both leave the {t(a)}", f"both now work in the {t(b)}"]}
        r = self._person([x for x, lb in w.res_lab.items() if lb])
        a = w.res_lab[r]
        b = self._other_lab(a, variant)
        return {"muts": [("res_lab", r, b)], "mentioned": [r, a, b],
                "facts": [f"{t(r)} leaves the {t(a)}", f"{r} now works in the {t(b)}"]}

    def u_join(self, variant: str) -> dict:
        b = self.rng.choice(list(self.w.lab_univ))
        if variant == "new_entity":
            r = self.nm.person()
            return {"muts": [("res_lab", r, b)], "mentioned": [r, b], "new_entities": [r],
                    "facts": [f"{r} (a newly graduated researcher) starts working in the {self._t(b)}"]}
        r = self._pool("free_res")
        return {"muts": [("res_lab", r, b)], "mentioned": [r, b],
                "facts": [f"{self._t(r)}, an independent researcher with no lab, starts working in the {self._t(b)}"]}

    def u_leave(self, variant: str) -> dict:
        r = self._person([x for x, lb in self.w.res_lab.items() if lb])
        a = self.w.res_lab[r]
        how = self.rng.choice(["leaves academia", "ends the contract", "retires from research"])
        return {"muts": [("res_lab", r, None)], "mentioned": [r, a],
                "facts": [f"{self._t(r)} {how} and no longer works in the {self._t(a)}; no new lab"]}

    def u_director(self, variant: str) -> dict:
        w, t = self.w, self._t
        if variant == "appoint":
            lab = self.rng.choice([lb for lb, d in w.lab_director.items() if d is None])
            new = self._pool("free_profs", reuse=True)
            return {"muts": [("lab_director", lab, new)], "mentioned": [new, lab],
                    "facts": [f"the {t(lab)}, which had no director, is now directed by {t(new)}"]}
        lab = self.rng.choice([lb for lb, d in w.lab_director.items() if d])
        old = w.lab_director[lab]
        if variant == "leave":
            return {"muts": [("lab_director", lab, None)], "mentioned": [old, lab],
                    "facts": [f"{t(old)} steps down as director of the {t(lab)} and leaves; no successor yet"]}
        new = self._pool("free_profs", reuse=True)
        return {"muts": [("lab_director", lab, new)], "mentioned": [old, new, lab],
                "facts": [f"{t(old)} no longer directs the {t(lab)} and leaves", f"{t(new)} is the new director of the {lab}"]}

    def u_lab(self, variant: str) -> dict:
        w = self.w
        lab = self.rng.choice(list(w.lab_univ))
        old = w.lab_univ[lab]
        new = w.lab_univ[self._other_lab(lab, variant)]
        return {"muts": [("lab_univ", lab, new)], "mentioned": [lab, old, new],
                "facts": [f"the whole {self._t(lab)} moves: it no longer belongs to {self._t(old)}",
                          f"the {lab} now belongs to {self._t(new)}; its director and researchers move with it"]}

    # ---- không có thay đổi -----------------------------------------------------------------------------------
    def no_change(self, variant: str) -> dict:
        w, t = self.w, self._t
        verb = self.rng.choice(_RUMOUR)
        if variant == "football":
            p = self._person([x for x, c in w.player_club.items() if c])
            a = w.player_club[p]
            b = self._other_club(a, "any")
        elif variant == "corporate":
            p = self._person([x for x, d in w.emp_dept.items() if d])
            a = w.emp_dept[p]
            b = self._other_dept(a, "other")
        else:
            p = self._person([x for x, lb in w.res_lab.items() if lb])
            a = w.res_lab[p]
            b = self._other_lab(a, "other")
        return {"muts": [], "mentioned": [p, a, b],
                "facts": [f"NOTHING HAS CHANGED. {t(p)} {verb} {t(b)}",
                          f"{p} is still with {t(a)}; make it unmistakable that no move has happened"]}


# ---------------------------------------------------------------------------------------------------------------
# Dựng mẫu: chênh lệch, nhãn độ sâu, đề bài
# ---------------------------------------------------------------------------------------------------------------


def _depths(changed: set[Triple], proof: dict[Triple, tuple[Triple, ...]]) -> dict[Triple, int]:
    """Độ sâu của từng triple bị đổi: 0 cho fact gốc; 1 + độ sâu lớn nhất của các tiền đề cũng bị đổi."""
    memo: dict[Triple, int] = {}

    def depth(tr: Triple) -> int:
        if tr not in memo:
            moved = [q for q in proof[tr] if q in changed]
            memo[tr] = 1 + max(depth(q) for q in moved) if moved else 0
        return memo[tr]
    return {tr: depth(tr) for tr in changed}


def generate(seed: int = SEED, scale: float = 1.0) -> dict:
    """Toàn bộ bộ dữ liệu trừ bản tin: KG nền, danh sách mẫu (chênh lệch + nhãn) và đề bài cho người viết."""
    rng = random.Random(seed)
    style_rng = random.Random(seed + 1)                           # gợi ý văn phong tách riêng: đổi gợi ý không đổi sự kiện
    world, names, pools = build_world(rng)
    before = closure(world)
    sampler = Sampler(world, names, pools, rng)
    plan = [(f, v) for f, v, n in PLAN for _ in range(max(1, round(n * scale)))]
    rng.shuffle(plan)
    rows, briefs = [], []
    for idx, (family, variant) in enumerate(plan):
        ev = sampler.make(family, variant)
        after = closure(world.mutate(ev["muts"])) if ev["muts"] else before
        removed, added = set(before) - set(after), set(after) - set(before)
        d_del, d_add = _depths(removed, before), _depths(added, after)
        mentioned = set(ev["mentioned"])
        direct = [["INVALIDATE", *tr] for tr in sorted(removed) if d_del[tr] == 0] + \
                 [["ADD", *tr] for tr in sorted(added) if d_add[tr] == 0]
        indirect = [["INVALIDATE", *tr, d_del[tr]] for tr in sorted(removed) if d_del[tr] > 0] + \
                   [["ADD", *tr, d_add[tr]] for tr in sorted(added) if d_add[tr] > 0]
        # thay đổi trực tiếp đúng bằng các cạnh đổi có cả hai đầu được nhắc trong bản tin (quy ước của GoldExplicitExtractor)
        both = {tr for tr in removed | added if tr[0] in mentioned and tr[2] in mentioned}
        assert both == {tuple(op[1:]) for op in direct}, (family, variant, ev)
        assert all(tr[1] in DERIVED for tr in removed | added if tr not in both), (family, variant)
        domain = {"f": "football", "c": "corporate", "u": "university"}.get(family[0], variant)
        rows.append({
            "idx": idx, "event": f"{family}/{variant}", "season": "", "text": "", "mentioned": ev["mentioned"],
            "after_add": [list(tr) for tr in sorted(added)], "after_del": [list(tr) for tr in sorted(removed)],
            "extra": {"domain": domain, "family": family, "variant": variant,
                      "max_depth": max([op[-1] for op in indirect], default=0),
                      "direct": direct, "indirect": indirect, "new_entities": ev["new_entities"]},
        })
        briefs.append({
            "idx": idx, "domain": domain, "no_change": family == "no_change",
            "style": style_rng.choice(STYLES), "sentences": style_rng.choice([1, 1, 2, 2, 2, 3, 3, 4]),
            "extras": style_rng.sample(EXTRAS_RUMOUR if family == "no_change" else EXTRAS, style_rng.choice([0, 1, 1, 2])),
            "facts": ev["facts"], "must_mention": ev["mentioned"],
        })
    all_names = sorted({e for h, _, t in before for e in (h, t)} | {e for r in rows for e in r["extra"]["new_entities"]})
    return {"world": world, "base": sorted(before), "rows": rows, "briefs": briefs, "names": all_names,
            "relations": sorted({r for _, r, _ in before})}


# ---------------------------------------------------------------------------------------------------------------
# Kiểm bản tin
# ---------------------------------------------------------------------------------------------------------------

MIN_WORDS, MAX_WORDS = 10, 130


def check_text(text: str, must: list[str], name_index: dict[tuple[str, ...], str]) -> list[str]:
    """Lỗi của một bản tin: thiếu thực thể phải nhắc, nhắc thực thể khác của KG, quá ngắn / quá dài, có xuống dòng."""
    errors = []
    if "\n" in text.strip():
        errors.append("có xuống dòng")
    n_words = len(text.split())
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        errors.append(f"{n_words} từ (cần {MIN_WORDS}–{MAX_WORDS})")
    toks = _norm_tokens(text)
    longest = max(len(k) for k in name_index)
    found: set[str] = set()
    for size in range(1, longest + 1):
        for i in range(len(toks) - size + 1):
            hit = name_index.get(tuple(toks[i:i + size]))
            if hit:
                found.add(hit)
    errors += [f"thiếu tên: {m}" for m in must if m not in found]
    errors += [f"nhắc thực thể ngoài đề bài: {x}" for x in sorted(found - set(must))]
    return errors


def load_texts(path: Path) -> dict[int, str]:
    """Bản tin từ một tệp JSONL {"idx", "text"} hoặc mọi tệp *.jsonl trong một thư mục."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out: dict[int, str] = {}
    for f in files:
        for line in f.read_text(encoding="utf8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[int(d["idx"])] = str(d["text"]).strip()
    return out


def check_all(data: dict, texts: dict[int, str]) -> dict[int, list[str]]:
    index = {tuple(n.split()): n for n in data["names"]}
    bad: dict[int, list[str]] = {}
    for row in data["rows"]:
        i = row["idx"]
        errs = ["chưa có bản tin"] if i not in texts else check_text(texts[i], row["mentioned"], index)
        if errs:
            bad[i] = errs
    return bad


# ---------------------------------------------------------------------------------------------------------------
# Thống kê + CLI
# ---------------------------------------------------------------------------------------------------------------


def stats(data: dict) -> dict:
    rows = data["rows"]
    by_depth: Counter = Counter()
    for r in rows:
        for op in r["extra"]["indirect"]:
            by_depth[f"{op[0]}@{op[-1]}"] += 1
    n_ind = sum(len(r["extra"]["indirect"]) for r in rows)
    return {
        "name": "KGU-Synth", "seed": SEED, "n": len(rows),
        "base_triples": len(data["base"]), "base_entities": len({e for h, _, t in data["base"] for e in (h, t)}),
        "relations": len(data["relations"]),
        "direct_ops": sum(len(r["extra"]["direct"]) for r in rows), "indirect_ops": n_ind,
        "indirect_per_sample": round(n_ind / len(rows), 2),
        "indirect_by_kind_depth": dict(sorted(by_depth.items())),
        "samples_by_max_depth": dict(sorted(Counter(r["extra"]["max_depth"] for r in rows).items())),
        "samples_by_domain": dict(sorted(Counter(r["extra"]["domain"] for r in rows).items())),
        "samples_by_event": dict(sorted(Counter(r["event"] for r in rows).items())),
        "samples_with_new_entity": sum(1 for r in rows if r["extra"]["new_entities"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["briefs", "check", "build", "stats"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale", type=float, default=1.0, help="hệ số nhân số mẫu của mỗi loại sự kiện (1.0 = 1.000 mẫu)")
    ap.add_argument("--out-dir", type=Path, help="briefs: thư mục ghi đề bài theo lô")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--texts", type=Path, help="check/build: tệp JSONL {idx, text} hoặc thư mục chứa các tệp đó")
    ap.add_argument("--out", type=Path, default=Path("data/synth/kgu_synth.jsonl"))
    ap.add_argument("--placeholder", action="store_true", help="build: dùng các câu của đề bài làm bản tin (chỉ để thử mã)")
    args = ap.parse_args()
    data = generate(args.seed, args.scale)
    assert not nested_names(data["names"]), nested_names(data["names"])[:5]
    if args.cmd == "stats":
        print(json.dumps(stats(data), indent=2, ensure_ascii=False))
    elif args.cmd == "briefs":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        size = -(-len(data["briefs"]) // args.chunks)
        for c in range(args.chunks):
            part = data["briefs"][c * size:(c + 1) * size]
            (args.out_dir / f"briefs_{c:02d}.jsonl").write_text(
                "".join(json.dumps(b, ensure_ascii=False) + "\n" for b in part), encoding="utf8")
        print(f"{len(data['briefs'])} đề bài -> {args.out_dir} ({args.chunks} lô, {size} đề mỗi lô)")
    elif args.cmd == "check":
        texts = load_texts(args.texts)
        bad = check_all(data, texts)
        for i, errs in sorted(bad.items()):
            if errs != ["chưa có bản tin"]:
                print(f"idx {i}: " + "; ".join(errs))
        missing = sum(1 for e in bad.values() if e == ["chưa có bản tin"])
        print(f"{len(texts)} bản tin, {len(bad) - missing} lỗi, {missing} mẫu chưa có bản tin")
        sys.exit(1 if bad else 0)
    else:
        if args.placeholder:
            texts = {b["idx"]: ". ".join(b["facts"]) + "." for b in data["briefs"]}
        else:
            texts = load_texts(args.texts)
            bad = check_all(data, texts)
            if bad:
                raise SystemExit(f"{len(bad)} bản tin lỗi hoặc thiếu; chạy lệnh check trước")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        head = {"base": [list(tr) for tr in data["base"]], "cumulative": False, "relations": data["relations"],
                "meta": {"name": "KGU-Synth", "seed": args.seed, "generator": "scripts/gen_synth.py"},
                "entity_types": data["world"].etype}
        with args.out.open("w", encoding="utf8") as f:
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for row in data["rows"]:
                f.write(json.dumps({**row, "text": texts[row["idx"]]}, ensure_ascii=False) + "\n")
        st = stats(data)
        args.out.with_suffix(".stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(st, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
