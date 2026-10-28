"""Runner chung cho kernel kgu-covid-repeat-*: chạy lại 2020-COVID-NYT (đủ 122 nhóm ngày) cho bốn hệ đã có lượt 1, để đo
độ ổn định giữa các lượt như paper ATOM (Jaccard tập 5-tuple giữa lượt 1 và lượt r; scripts/covid_stability.py) và
khoảng dao động F1. Cấu hình y như lượt 1:
- mine_v2: scripts/run_covid_nyt.py --condense --canon-every 10 --tau 0.75 (như kgu-covid-condense-canon, bản v2);
- atom, atom_direct: scripts/atom_covid.py --mode atom / direct, itext2kg 1.1.0 trong venv Python 3.11 (như
  kgu-home-benchmarks);
- kggen: scripts/kggen_covid.py, kg-gen 0.4.0 cài vào python hệ thống SAU khi các hệ khác chạy xong (như
  kgu-home-benchmarks, để gói của kg-gen không đổi môi trường của các hệ kia).
Lượt 1: results/kaggle/covid_v2/v2.jsonl, results/kaggle/home_bench/covid/{atom,atom-direct,kggen}.jsonl.
Gemma 4 E4B qua llama-server CUDA (GPU 0), temperature 0; embedder gộp tên và chấm all-mpnet-base-v2 (GPU 1).

Stub (kaggle/covid_repeat/<lượt>/kernel.py) đặt COVID_REPEAT (vd. "r2") và COVID_SYSTEMS (mặc định cả bốn).
Đầu ra /kaggle/working/out/: <hệ>.<lượt>.jsonl (+ .kg.json ... của mine_v2), scores_covid.json (lượt này và lượt 1
chấm lại cùng bộ chấm), summary.json, logs/*.log, server.log.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

REPEAT = os.environ.get("COVID_REPEAT", "r2")
SYSTEMS = [s for s in os.environ.get("COVID_SYSTEMS", "mine_v2,atom,atom_direct,kggen").split(",") if s]
LIMIT = int(os.environ["COVID_LIMIT"]) if os.environ.get("COVID_LIMIT") else None
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
RUN1 = {"mine_v2": "results/kaggle/covid_v2/v2.jsonl", "atom": "results/kaggle/home_bench/covid/atom.jsonl",
        "atom_direct": "results/kaggle/home_bench/covid/atom-direct.jsonl",
        "kggen": "results/kaggle/home_bench/covid/kggen.jsonl"}
T0 = time.time()


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


for d in (OUT, LOGS):
    os.makedirs(d, exist_ok=True)
summary: dict = {"repeat": REPEAT, "systems": SYSTEMS, "limit": LIMIT, "runs": {}, "notes": []}

# 1. GPU, code, gói, dữ liệu
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
if not os.path.exists(KLTN):
    sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser 'openai>=1'")
rows = sorted((json.loads(line) for line in open(f"{KLTN}/data/covid_nyt/2020_covid_nyt.jsonl", encoding="utf-8")
               if line.strip()), key=lambda r: (r["date"], r["row"]))[:LIMIT]
COVID = f"{WORK}/covid.jsonl"
with open(COVID, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary["covid_n"] = len(rows)

# 2. llama-server, ghim GPU 0
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


def timed(run: str, cmd: str, **kw) -> bool:
    t0 = time.time()
    try:
        sh(f"{cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0", executable="/bin/bash", **kw)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    summary["runs"][run] = {"ok": ok, "seconds": round(time.time() - t0, 1)}
    print(f"=== {run}: ok={ok} {summary['runs'][run]['seconds']}s, tổng {round((time.time() - T0) / 60)} phút", flush=True)
    return ok


def out_path(system: str) -> str:
    return f"{OUT}/{system}.{REPEAT}.jsonl"


# 3. các hệ (kggen cuối cùng: cài kg-gen đổi gói của python hệ thống)
emb_device = "cuda:1" if len(gpus) >= 2 else "cuda:0"
if "mine_v2" in SYSTEMS:
    timed("mine_v2", f"python -u {KLTN}/scripts/run_covid_nyt.py --data {COVID} --out {out_path('mine_v2')} --condense "
                     f"--canon-every {CANON_EVERY} --tau {TAU} --device {emb_device}", env=env)
if {"atom", "atom_direct"} & set(SYSTEMS):
    venv = f"{WORK}/venv_atom"
    try:
        sh(f"pip install -q uv && uv venv -q --python 3.11 {venv} && "
           f"uv pip install -q --python {venv}/bin/python itext2kg==1.1.0 dateparser")
        py_atom = f"{venv}/bin/python"
    except subprocess.CalledProcessError:
        summary["notes"].append("venv atom lỗi → cài itext2kg vào python hệ thống")
        sh("pip install -q itext2kg==1.1.0")
        py_atom = "python"
    for system, mode in (("atom", "atom"), ("atom_direct", "direct")):
        if system in SYSTEMS:
            timed(system, f"{py_atom} -u -W ignore {KLTN}/scripts/atom_covid.py --mode {mode} --data {COVID} "
                          f"--out {out_path(system)}", env=env)
if "kggen" in SYSTEMS:
    sh("pip install -q kg-gen==0.4.0")
    timed("kggen", f"python -u {KLTN}/scripts/kggen_covid.py --data {COVID} --out {out_path('kggen')}", env=env)
server.kill()

# 4. chấm lượt này + lượt 1 bằng cùng bộ chấm, rồi độ ổn định lượt 1 ↔ lượt này
done = [s for s in SYSTEMS if os.path.exists(out_path(s))]
preds = " ".join(f"--pred {s}.{REPEAT}={out_path(s)}" for s in done)
preds += " " + " ".join(f"--pred {s}.r1={KLTN}/{RUN1[s]}" for s in done if not LIMIT)
try:
    sh(f"set -o pipefail; python -u {KLTN}/scripts/score_covid_nyt.py --data {COVID} {preds} "
       f"--embedder {SCORE_EMBEDDER} --threshold {SCORE_THRESHOLD} --device cuda "
       f"--out {OUT}/scores_covid.json 2>&1 | tee {LOGS}/score.log", executable="/bin/bash", env=env)
    summary["scores"] = json.load(open(f"{OUT}/scores_covid.json"))["results"][SCORE_EMBEDDER][SCORE_THRESHOLD]
except subprocess.CalledProcessError:
    print("!!! chấm lỗi, xem logs/score.log", flush=True)
if not LIMIT:
    runs = " ".join(f"--run {s}=r1:{KLTN}/{RUN1[s]},{REPEAT}:{out_path(s)}" for s in done)
    try:
        sh(f"set -o pipefail; python -u {KLTN}/scripts/covid_stability.py --data {COVID} {runs} --released --embedder {SCORE_EMBEDDER} "
           f"--threshold {SCORE_THRESHOLD} --device cuda --out {OUT}/stability.json 2>&1 | tee {LOGS}/stability.log",
           executable="/bin/bash", env=env)
        summary["stability"] = json.load(open(f"{OUT}/stability.json"))
    except subprocess.CalledProcessError:
        print("!!! độ ổn định lỗi, xem logs/stability.log", flush=True)

for s in done:
    rs = [json.loads(line) for line in open(out_path(s), encoding="utf-8") if line.strip()]
    summary.setdefault("outputs", {})[s] = {
        "rows": len(rs), "empty": sum(not r["quintuples"] for r in rs),
        "avg_quintuples": round(sum(len(r["quintuples"]) for r in rs) / max(len(rs), 1), 2)}
summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
for d in (KLTN, f"{WORK}/models", f"{WORK}/llama-bin", f"{WORK}/venv_atom"):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model/venv vào output
print("DONE", flush=True)
