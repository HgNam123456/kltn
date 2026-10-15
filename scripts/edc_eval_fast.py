"""Chạy evaluate/evaluation_script.py của EDC y nguyên, chỉ thay bước duyệt mọi hoán vị trong calculateSystemScore.
Script gốc tự đệm "" cho số triple dự đoán và gold bằng nhau rồi thử mọi hoán vị ghép (n! với n = max(số dự đoán,
số gold): câu 10 triple = 3,6 triệu, câu 12 triple = 479 triệu) và giữ hoán vị ĐẦU TIÊN theo thứ tự của
itertools.permutations (thứ tự từ điển) có tổng điểm lớn nhất. Ở đây: n ≤ EXACT_N vẫn duyệt hết như gốc; n lớn hơn tìm
đúng hoán vị đó bằng gán tối ưu (Hungarian): giá trị tối ưu rồi chọn tham lam từng vị trí cột nhỏ nhất còn đạt tối ưu.
Phần còn lại của script (điểm từng cặp, tổng hợp, in kết quả) không đổi.
Kiểm tương đương: output khớp output của script gốc trên WebNLG và REBEL chạy thử (kaggle/edc_eval_check).

Dùng (cwd = thư mục repo EDC, như script gốc):
  python scripts/edc_eval_fast.py --edc_script evaluate/evaluation_script.py --edc_output cand.txt --reference ref.txt
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

EXACT_N = 7
OLD = """            choosescore = list(
                itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newcandlist[idx]))
            )
            choosescore = [list(x) for x in choosescore]
"""
NEW = """            choosescore = [_best_perm(totalsemevallist[idx])]
"""


def _assign_value(M: list[list[float]], rows: list[int], cols: list[int]) -> float:
    """Tổng lớn nhất của phép gán một-một rows ↔ cols (|rows| = |cols|) trên M — Hungarian O(k³) có thế vị."""
    k = len(rows)
    if k == 0:
        return 0.0
    inf = float("inf")
    a = [[-M[r][c] for c in cols] for r in rows]
    u, v, p, way = [0.0] * (k + 1), [0.0] * (k + 1), [0] * (k + 1), [0] * (k + 1)
    for i in range(1, k + 1):
        p[0], j0 = i, 0
        minv, used = [inf] * (k + 1), [False] * (k + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, k + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sum(M[rows[p[j] - 1]][cols[j - 1]] for j in range(1, k + 1))


def best_perm(M: list[list[float]]) -> list[int]:
    """Hoán vị đầu tiên (thứ tự từ điển) có tổng M[i][perm[i]] lớn nhất — đúng hoán vị script gốc giữ lại."""
    n = len(M)
    if n <= EXACT_N:                                   # duyệt hết y như gốc (cộng dồn cùng thứ tự, so sánh ">")
        best, best_s = None, None
        for perm in itertools.permutations(range(n), n):
            s = 0
            for i, j in enumerate(perm):
                s += M[i][j]
            if best is None or s > best_s:
                best, best_s = list(perm), s
        return best
    cols = list(range(n))
    target = _assign_value(M, list(range(n)), cols)
    perm: list[int] = []
    for i in range(n):
        rest_rows = list(range(i + 1, n))
        for j in cols:                                 # cols luôn tăng dần → thử cột nhỏ nhất trước
            rest = [c for c in cols if c != j]
            if M[i][j] + _assign_value(M, rest_rows, rest) >= target - 1e-9:
                perm.append(j)
                target -= M[i][j]
                cols = rest
                break
    return perm


def _best_perm(row_scores) -> list[int]:
    """row_scores = totalsemevallist[idx]: điểm của script gốc cho từng cặp (dự đoán i, gold j); điểm ghép = trung
    bình 4 F1 như calculateSystemScore."""
    M = [[statistics.mean([c["ent_type"]["f1"], c["partial"]["f1"], c["strict"]["f1"], c["exact"]["f1"]])
          for c in row] for row in row_scores]
    return best_perm(M)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--edc_script", required=True)
    args, rest = ap.parse_known_args()
    src = Path(args.edc_script).read_text(encoding="utf8")
    assert src.count(OLD) == 1, "không tìm thấy đoạn duyệt hoán vị trong script chấm của EDC"
    sys.argv = [args.edc_script, *rest]
    exec(compile(src.replace(OLD, NEW), args.edc_script, "exec"),
         {"__name__": "__main__", "__file__": args.edc_script, "_best_perm": _best_perm})


if __name__ == "__main__":
    main()
