"""Kaggle kernel (CPU): kiểm scripts/edc_eval_fast.py cho ra ĐÚNG output của evaluate/evaluation_script.py gốc của EDC
(EDC_COMMIT). So toàn văn stdout (kể cả số Correct/Spurious/... từng loại) với output script gốc đã lưu trong repo:
WebNLG đủ 1.165 câu của kernel kgu-home-benchmarks (EDC và mình) và REBEL/Wiki-NRE 30 câu của lượt chạy thử
kgu-edc-rebel-wikinre. Đầu ra /kaggle/working/out/: <tên>.txt (stdout bản nhanh), summary.json.
"""
import json
import os
import subprocess
import time

KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
EDC_REPO, EDC_COMMIT = "https://github.com/clear-nus/edc.git", "02aad17e2fc1e9845c84746b384e7fd3160edc1a"
WORK = "/kaggle/working"
OUT, KLTN, EDC = f"{WORK}/out", f"{WORK}/kltn", f"{WORK}/edc"


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


os.makedirs(OUT, exist_ok=True)
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
sh(f"git clone -q {EDC_REPO} {EDC} && git -C {EDC} checkout -q {EDC_COMMIT}")
sh("pip install -q nervaluate==0.1.8 unidecode beautifulsoup4 regex")
R = f"{KLTN}/results/kaggle"
CASES = [("webnlg_edc", f"{R}/home_bench/webnlg/edc_canon_kg.txt", "webnlg", None, f"{R}/home_bench/webnlg/edc_eval.txt"),
         ("webnlg_mine", f"{R}/home_bench/webnlg/mine_canon_kg.txt", "webnlg", None, f"{R}/home_bench/webnlg/mine_eval.txt")]
for ds in ("rebel", "wiki-nre"):
    for s in ("edc", "mine"):
        CASES.append((f"{ds}_{s}", f"{R}/edc_bench_smoke/{ds}/{s}_canon_kg.txt", ds, 30,
                      f"{R}/edc_bench_smoke/{ds}/{s}_eval.txt"))

summary = {"kltn_commit": sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip(), "cases": {}}
for name, cand, ds, limit, saved in CASES:
    refs = open(f"{EDC}/evaluate/references/{ds}.txt", encoding="utf-8").read().splitlines()[:limit]
    ref = f"{WORK}/{name}_ref.txt"
    open(ref, "w", encoding="utf-8").write("\n".join(refs) + "\n")
    os.makedirs(f"{WORK}/cand/{name}", exist_ok=True)             # thư mục riêng: script ghi result_xmls/<thư mục cha>
    sh(f"cp {cand} {WORK}/cand/{name}/canon_kg.txt")
    t0 = time.time()
    got = sh(f"python {KLTN}/scripts/edc_eval_fast.py --edc_script evaluate/evaluation_script.py "
             f"--edc_output {WORK}/cand/{name}/canon_kg.txt --reference {ref}", cwd=EDC, capture=True)
    sec = round(time.time() - t0, 1)
    open(f"{OUT}/{name}.txt", "w", encoding="utf-8").write(got)
    want = open(saved, encoding="utf-8").read()
    same = got.strip() == want.strip()
    diff = [(a, b) for a, b in zip(got.strip().splitlines(), want.strip().splitlines()) if a != b][:5]
    summary["cases"][name] = {"identical": same, "seconds": sec, "first_diffs": diff}
    print(f"=== {name}: identical={same} {sec}s {diff}", flush=True)
summary["all_identical"] = all(c["identical"] for c in summary["cases"].values())
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
