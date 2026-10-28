"""Kernel kgu-quality: hai thước đo không cần gold trên output ĐÃ CÓ trong repo (không chạy lại hệ nào):
- độ chính xác theo văn bản nguồn: scripts/judge_support.py, judge Gemma 4 E4B qua llama-server (GPU 0);
- độ gọn: scripts/kg_redundancy.py, all-mpnet-base-v2 (COVID-NYT: KG tích lũy cả 122 nhóm; MINE-1: từng bài).
Output: COVID-NYT lượt 1 (results/kaggle/covid_v2/v2.jsonl của mình, results/kaggle/home_bench/covid/ của ATOM,
ATOM trích thẳng, KGGen) + output GPT-4.1 tác giả phát hành (cột calib lượt 1); MINE-1 (results/kaggle/mine/ của mình
và KGGen-Gemma) + ba KG phát hành trong dataset (ghép lại đúng bài).
Đầu ra /kaggle/working/out/: support.json (+ support.<hệ>.jsonl), redundancy.json, summary.json, logs/, server.log.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

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

COVID = {"covid/mine_v2": "covid:results/kaggle/covid_v2/v2.jsonl",
         "covid/atom": "covid:results/kaggle/home_bench/covid/atom.jsonl",
         "covid/atom_direct": "covid:results/kaggle/home_bench/covid/atom-direct.jsonl",
         "covid/kggen": "covid:results/kaggle/home_bench/covid/kggen.jsonl",
         "covid/released_gpt41_atom": "covid-released:quintuples_gpt41_from_factoids",
         "covid/released_gpt41_direct": "covid-released:quintuples_gpt41"}
MINE = {"mine/mine": "mine:results/kaggle/mine/mine.jsonl", "mine/kggen_gemma": "mine:results/kaggle/mine/kggen.jsonl",
        "mine/released_kggen": "mine-released:kggen", "mine/released_graphrag": "mine-released:graphrag_kg",
        "mine/released_openie": "mine-released:openie_kg"}


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
summary: dict = {"runs": {}}
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
if not os.path.exists(KLTN):
    sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser 'openai>=1'")
from huggingface_hub import hf_hub_download  # noqa: E402

mine_data = hf_hub_download(MINE_REPO, MINE_FILE, repo_type="dataset", local_dir=f"{WORK}/mine")
covid_data = f"{KLTN}/data/covid_nyt/2020_covid_nyt.jsonl"

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
env = {**os.environ, "PYTHONIOENCODING": "utf8", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": KLTN,
       "KGU_LLM_BASE_URL": f"{SERVER}/v1", "KGU_LLM_MODEL": "local", "KGU_LLM_API_KEY": "none",
       "KGU_LLM_EXTRA_BODY": "{}", "KGU_LLM_MAX_TOKENS": "4096"}


def step(name: str, cmd: str) -> None:
    t0 = time.time()
    try:
        sh(f"set -o pipefail; cd {KLTN} && {cmd} 2>&1 | tee {LOGS}/{name}.log", executable="/bin/bash", env=env)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {name} lỗi, xem logs/{name}.log", flush=True)
        ok = False
    summary["runs"][name] = {"ok": ok, "seconds": round(time.time() - t0, 1)}


def specs(d: dict) -> str:
    return " ".join(f"--system {k}={v}" for k, v in d.items())


step("judge", f"python -u scripts/judge_support.py {specs(COVID)} {specs(MINE)} --covid-data {covid_data} "
              f"--mine-data {mine_data} --workers {WORKERS} --out {OUT}/support.json")
server.kill()
step("redundancy_covid", f"python -u scripts/kg_redundancy.py {specs(COVID)} --covid-data {covid_data} --scope global "
                         f"--device cuda --out {OUT}/redundancy.json")
step("redundancy_mine", f"python -u scripts/kg_redundancy.py {specs(MINE)} --mine-data {mine_data} --scope doc "
                        f"--device cuda --out {OUT}/redundancy.json")
for name in ("support", "redundancy"):
    if os.path.exists(f"{OUT}/{name}.json"):
        summary[name] = json.load(open(f"{OUT}/{name}.json"))
summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
for d in (KLTN, f"{WORK}/models", f"{WORK}/llama-bin", f"{WORK}/mine"):
    shutil.rmtree(d, ignore_errors=True)
print("DONE", flush=True)
