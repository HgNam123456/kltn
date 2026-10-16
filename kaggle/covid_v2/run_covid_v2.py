"""Kaggle kernel: ablation trên 2020-COVID-NYT cho pipeline của mình (spec
docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md): v0 = TemporalKG hiện tại, v1 = + rút câu đơn và
trích bổ sung, v2 = v1 + gộp tên k-means/LLM mỗi CANON_EVERY nhóm ngày. Gemma 4 E4B qua llama-server CUDA (GPU 0) như
kgu-home-benchmarks; embedder gộp tên all-mpnet-base-v2 trên GPU 1. Chấm scripts/score_covid_nyt.py (mọi nhóm + held-out
bỏ DEV_N nhóm đầu) và scripts/score_covid_merge.py (chỉ số gộp tên của ATOM trên KG cuối).
LIMIT = số nhóm ngày theo thứ tự ngày (20 = tập dev để chỉnh prompt/τ; None = đủ 122). VARIANTS chọn biến thể.
Đầu ra /kaggle/working/out/: <v>.jsonl (+ .kg.json, .aliases.json; v2 thêm .merges.jsonl, .canon.json),
scores_covid[_heldout].json, merge_scores.json, summary.json, logs/*.log, server.log.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

LIMIT = None
VARIANTS = ["v0", "v1", "v2"]
DEV_N = 20
CANON_EVERY, TAU = 10, 0.75
WORKERS = 4
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
SCORE_EMBEDDER, SCORE_THRESHOLD = "sentence-transformers/all-mpnet-base-v2", "0.7"
WORK = "/kaggle/working"
OUT, LOGS = f"{WORK}/out", f"{WORK}/out/logs"
KLTN = f"{WORK}/kltn"
SERVER = "http://127.0.0.1:8080"
T0 = time.time()


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


for d in (OUT, LOGS):
    os.makedirs(d, exist_ok=True)
summary: dict = {"limit": LIMIT, "variants": VARIANTS, "canon_every": CANON_EVERY, "tau": TAU, "runs": {},
                 "outputs": {}, "canon": {}, "notes": []}

# 1. GPU, code, gói
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser 'openai>=1'")

# 2. dữ liệu theo thứ tự ngày
rows = sorted(read_jsonl(f"{KLTN}/data/covid_nyt/2020_covid_nyt.jsonl"), key=lambda r: (r["date"], r["row"]))[:LIMIT]
COVID, COVID_HELDOUT = f"{WORK}/covid.jsonl", f"{WORK}/covid_heldout.jsonl"
for path, part in ((COVID, rows), (COVID_HELDOUT, rows[DEV_N:])):
    with open(path, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary["covid_n"], summary["heldout_n"] = len(rows), max(len(rows) - DEV_N, 0)

# 3. llama-server (giống kaggle/home_bench/run_home_bench.py), ghim GPU 0
from huggingface_hub import hf_hub_download  # noqa: E402

cached = find_dir("llama-server")
assert cached, "thiếu dataset ngocnam2005/llama-server-cuda"
bin_dir = f"{WORK}/llama-bin"
shutil.copytree(cached, bin_dir, dirs_exist_ok=True)
sh(f"chmod +x {bin_dir}/llama-server")
model = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=f"{WORK}/models")
draft = hf_hub_download(GGUF_REPO, MTP_FILE, local_dir=f"{WORK}/models")
COMMON = f"{bin_dir}/llama-server -m {model} --alias local --host 127.0.0.1 --port 8080 -ngl 99 --jinja --temp 0"
ATTEMPTS = [
    COMMON + f" --parallel {WORKERS} --cont-batching --kv-unified-per-slot 8192 --reasoning off --flash-attn on"
             f" --spec-type draft-mtp --spec-draft-n-max 2 --model-draft {draft}",
    COMMON + f" --parallel {WORKERS} --cont-batching -c {8192 * WORKERS} --flash-attn on",
]


def healthy() -> bool:
    try:
        return b"ok" in urllib.request.urlopen(f"{SERVER}/health", timeout=3).read()
    except Exception:
        return False


server = None
for cmd in ATTEMPTS:
    print(f"$ {cmd}", flush=True)
    server = subprocess.Popen(cmd, shell=True, stdout=open(f"{OUT}/server.log", "ab"), stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    for _ in range(180):
        if healthy() or server.poll() is not None:
            break
        time.sleep(2)
    if healthy():
        break
    print("server không lên, thử bộ cờ khác", flush=True)
    server.kill()
assert healthy(), "llama-server không khởi động được, xem server.log"
summary["server_flags"] = cmd.replace(COMMON, "")

env = {**os.environ, "PYTHONIOENCODING": "utf8", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": KLTN,
       "KGU_LLM_BASE_URL": f"{SERVER}/v1", "KGU_LLM_MODEL": "local", "KGU_LLM_API_KEY": "none",
       "KGU_LLM_EXTRA_BODY": "{}", "KGU_LLM_MAX_TOKENS": "4096"}


def timed(run: str, cmd: str, n: int, **kw) -> bool:
    t0 = time.time()
    try:
        sh(f"{cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0", executable="/bin/bash", **kw)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    sec = round(time.time() - t0, 1)
    summary["runs"][run] = {"ok": ok, "seconds": sec, "sec_per_item": round(sec / max(n, 1), 2)}
    print(f"=== {run}: ok={ok} {sec}s, tổng {round((time.time() - T0) / 60)} phút", flush=True)
    return ok


# 4. biến thể
emb_device = "cuda:1" if len(gpus) >= 2 else "cuda:0"
FLAGS = {"v0": "", "v1": "--condense",
         "v2": f"--condense --canon-every {CANON_EVERY} --tau {TAU} --device {emb_device}"}
for v in VARIANTS:
    timed(v, f"python -u {KLTN}/scripts/run_covid_nyt.py --data {COVID} --out {OUT}/{v}.jsonl {FLAGS[v]}",
          len(rows), env=env)
server.kill()

# 5. chấm: trích xuất (mọi nhóm + held-out) và gộp tên
done = [v for v in VARIANTS if os.path.exists(f"{OUT}/{v}.jsonl")]
preds = " ".join(f"--pred {v}={OUT}/{v}.jsonl" for v in done)
for split, data in (("covid", COVID), ("covid_heldout", COVID_HELDOUT)):
    if not os.path.getsize(data):
        continue
    try:
        sh(f"set -o pipefail; python -u {KLTN}/scripts/score_covid_nyt.py --data {data} {preds} "
           f"--embedder {SCORE_EMBEDDER} --threshold {SCORE_THRESHOLD} --device cuda "
           f"--out {OUT}/scores_{split}.json 2>&1 | tee {LOGS}/score_{split}.log", executable="/bin/bash", env=env)
        summary[split] = json.load(open(f"{OUT}/scores_{split}.json"))["results"][SCORE_EMBEDDER][SCORE_THRESHOLD]
    except subprocess.CalledProcessError:
        print(f"!!! chấm {split} lỗi, xem logs/score_{split}.log", flush=True)
kgs = " ".join(f"--kg {v}={OUT}/{v}.jsonl.kg.json" for v in done if os.path.exists(f"{OUT}/{v}.jsonl.kg.json"))
try:
    sh(f"set -o pipefail; python -u {KLTN}/scripts/score_covid_merge.py --data {COVID} {kgs} "
       f"--embedder {SCORE_EMBEDDER} --device cuda --out {OUT}/merge_scores.json 2>&1 | tee {LOGS}/score_merge.log",
       executable="/bin/bash", env=env)
    summary["merge"] = json.load(open(f"{OUT}/merge_scores.json"))
except subprocess.CalledProcessError:
    print("!!! chấm gộp tên lỗi, xem logs/score_merge.log", flush=True)

# 6. thống kê output
for v in done:
    rs = read_jsonl(f"{OUT}/{v}.jsonl")
    n_q = sum(len(r["quintuples"]) for r in rs)
    summary["outputs"][v] = {
        "rows": len(rs), "empty": sum(not r["quintuples"] for r in rs), "avg_quintuples": round(n_q / max(len(rs), 1), 2),
        "calls": sum(r.get("calls", 0) for r in rs),
        "avg_sentences": round(sum(r.get("sentences", 0) for r in rs) / max(len(rs), 1), 2),
        "missing_before": sum(r.get("missing_before", 0) for r in rs),
        "missing_after": sum(r.get("missing_after", 0) for r in rs),
        "refill_sentences": sum(r.get("refill_sentences", 0) for r in rs),
        "errors": {k: sum(k in r for r in rs) for k in ("condense_error", "fill_error", "refill_error")}}
    if os.path.exists(f"{OUT}/{v}.jsonl.canon.json"):
        summary["canon"][v] = json.load(open(f"{OUT}/{v}.jsonl.canon.json"))
summary["canon_errors"] = {v: sum("error" in r for r in summary["canon"].get(v, [])) for v in done}

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
for d in (KLTN, f"{WORK}/models", f"{WORK}/llama-bin"):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model vào output
print("DONE", flush=True)
