"""Kaggle kernel: MINE-1 (benchmark của KGGen, HF josancamon/kg-gen-MINE-evaluation-dataset, 100 bài luận x 15 fact).
Dựng KG cho từng bài bằng (a) pipeline của mình (scripts/run_mine.py: TemporalKG rút câu đơn + gộp tên, như bản
COVID-NYT hiện tại) và (b) KGGen 0.4.0 (scripts/kggen_mine.py), cả hai bằng Gemma 4 E4B qua llama-server CUDA (GPU 0);
rồi chấm (scripts/score_mine.py) bằng đúng hàm truy xuất của kg-gen (all-MiniLM-L6-v2, k=8, sâu 2) với judge Gemma
thay cho gpt-5 của bản gốc. Chấm lại cả ba KG phát hành trong dataset (KGGen, GraphRAG, OpenIE của paper) để đo độ lệch
do đổi judge so với cột *_accuracy của dataset.
LIMIT = số bài (None = đủ). KGGEN_CLUSTER bật bước gom cụm của KGGen: tắt — lượt 1 (17/10) gom cụm bằng Gemma lặp mãi
vì Gemma trả nhãn ngoài danh sách cho phép, 6 bài sau ~10 giờ, kernel bị cắt ở 12 giờ; các lượt KGGen trên COVID-NYT và
EMERGE cũng không gom cụm. KG của mình đã dựng xong ở lượt 1 (results/kaggle/mine/mine.jsonl) → dùng lại (REUSE_MINE).
Mỗi bước có TIMEOUT để một bước treo không ăn hết 12 giờ.
Đầu ra /kaggle/working/out/: mine.jsonl, kggen.jsonl, scores.json (+ scores.<hệ>.jsonl từng fact), summary.json, logs/.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

LIMIT = None
KGGEN_CLUSTER = False
REUSE_MINE = True
# Lượt 3: KG của mình (lượt 1) và KGGen (lượt 2) đã có trong repo, điểm của hai bộ này từ lượt 2 đúng → chỉ chấm lại 3 KG
# phát hành sau khi ghép lại đúng bài (dataset lệch hàng ở ~80% hàng, xem scripts/score_mine.py:realign).
REUSE_KGGEN = True
SCORE_BUILT = False
TIMEOUT = {"mine": "3h", "kggen": "2h", "score": "5h"}
WORKERS = 4
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
MINE_REPO, MINE_FILE = "josancamon/kg-gen-MINE-evaluation-dataset", "data/train-00000-of-00001.parquet"
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


for d in (OUT, LOGS):
    os.makedirs(d, exist_ok=True)
summary: dict = {"limit": LIMIT, "kggen_cluster": KGGEN_CLUSTER, "runs": {}, "notes": []}

# 1. GPU, code, gói, dữ liệu
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser 'openai>=1' kg-gen==0.4.0")
from huggingface_hub import hf_hub_download  # noqa: E402

data = hf_hub_download(MINE_REPO, MINE_FILE, repo_type="dataset", local_dir=f"{WORK}/mine")
if LIMIT:
    import pandas as pd

    df = pd.read_parquet(data)
    df = df[df.essay_content.str.len() > 0].head(LIMIT)
    data = f"{WORK}/mine/limit.parquet"
    df.to_parquet(data)

# 2. llama-server (giống kaggle/covid_v2/run_covid_v2.py), ghim GPU 0
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


def timed(run: str, cmd: str) -> bool:
    t0 = time.time()
    try:
        sh(f"timeout {TIMEOUT[run]} {cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0",
           executable="/bin/bash", env=env)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    summary["runs"][run] = {"ok": ok, "seconds": round(time.time() - t0, 1)}
    print(f"=== {run}: ok={ok} {summary['runs'][run]['seconds']}s, tổng {round((time.time() - T0) / 60)} phút", flush=True)
    return ok


# 3. dựng KG: mình (embedder gộp tên trên GPU 1) và KGGen
emb_device = "cuda:1" if len(gpus) >= 2 else "cuda:0"
reused = f"{KLTN}/results/kaggle/mine/mine.jsonl"
if REUSE_MINE and not LIMIT and os.path.exists(reused):
    shutil.copy(reused, f"{OUT}/mine.jsonl")
    summary["notes"].append(f"mine.jsonl dùng lại từ lượt 1 ({reused}, commit {summary['kltn_commit'][:7]})")
else:
    timed("mine", f"python -u {KLTN}/scripts/run_mine.py --data {data} --out {OUT}/mine.jsonl --workers {WORKERS} "
                  f"--device {emb_device}")
reused_kggen = f"{KLTN}/results/kaggle/mine/kggen.jsonl"
if REUSE_KGGEN and not LIMIT and os.path.exists(reused_kggen):
    shutil.copy(reused_kggen, f"{OUT}/kggen.jsonl")
    summary["notes"].append(f"kggen.jsonl dùng lại từ lượt 2 ({reused_kggen})")
else:
    timed("kggen", f"python -u {KLTN}/scripts/kggen_mine.py --data {data} --out {OUT}/kggen.jsonl --workers {WORKERS}"
                   + (" --cluster" if KGGEN_CLUSTER else ""))

# 4. chấm: 2 KG mới + 3 KG phát hành, cùng judge Gemma
kgs = " ".join(f"--kg {n}={OUT}/{n}.jsonl" for n in ("mine", "kggen") if SCORE_BUILT and os.path.exists(f"{OUT}/{n}.jsonl"))
timed("score", f"python -u {KLTN}/scripts/score_mine.py --data {data} {kgs} --released kggen graphrag_kg openie_kg "
               f"--workers {WORKERS} --out {OUT}/scores.json")
server.kill()
if os.path.exists(f"{OUT}/scores.json"):
    summary["scores"] = json.load(open(f"{OUT}/scores.json"))

# 5. thống kê KG
for n in ("mine", "kggen"):
    path = f"{OUT}/{n}.jsonl"
    if os.path.exists(path):
        rs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        rel = [len((r.get("graph") or {}).get("relations", [])) for r in rs]
        ent = [len((r.get("graph") or {}).get("entities", [])) for r in rs]
        summary.setdefault("graphs", {})[n] = {
            "essays": len(rs), "empty": sum(x == 0 for x in rel), "errors": sum(bool("error" in r or r.get("errors")) for r in rs),
            "avg_relations": round(sum(rel) / max(len(rs), 1), 1), "avg_entities": round(sum(ent) / max(len(rs), 1), 1)}

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
for d in (KLTN, f"{WORK}/models", f"{WORK}/llama-bin", f"{WORK}/mine"):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model/dữ liệu vào output
print("DONE", flush=True)
