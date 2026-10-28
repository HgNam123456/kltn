"""Runner chung cho kernel kgu-extra-*: một benchmark trích triple theo schema ở định dạng EDC (thư mục có input.txt,
ref.txt, schema.csv, oie_few_shot_examples.txt, sd_few_shot_examples.txt — vd. NYT11-HRL từ scripts/prep_nyt11.py,
dataset Kaggle riêng tư) với các hệ đã dùng ở WebNLG/REBEL/Wiki-NRE, mọi hệ bằng Gemma 4 E4B qua llama-server CUDA:
- mine: scripts/run_webnlg.py (một lệnh gọi mỗi văn bản, quan hệ ép theo schema, few-shot OIE của bộ);
- edc: EDC gốc (clear-nus/edc @02aad17) OIE → Schema Definition → Canonicalization theo schema của bộ, không refinement;
- atom_edc: ATOM (itext2kg 1.1.0, scripts/atom_edc.py) thay bước OIE của EDC, rồi chuẩn hóa của EDC;
- kggen_edc: KGGen 0.4.0 (scripts/kggen_edc.py) thay bước OIE của EDC, rồi chuẩn hóa của EDC.
Sau khi chạy: độ chính xác theo văn bản nguồn (scripts/judge_support.py, judge Gemma), chấm gold (script EDC qua
scripts/edc_eval_fast.py; khớp chính xác kiểu NYT11 qua scripts/score_nyt11.py), độ gọn (scripts/kg_redundancy.py).
itext2kg và kg-gen mỗi gói một venv Python 3.11 riêng (itext2kg ghim numpy<2; kg-gen kéo gói mới) để không đổi môi
trường của EDC và pipeline của mình.

Stub (kaggle/extra_bench/<tên>/kernel.py) đặt EXTRA_DATASET (tên thư mục dữ liệu trong /kaggle/input, vd. "nyt11"),
EXTRA_SYSTEMS (tập con của mine,edc,atom_edc,kggen_edc), EXTRA_LIMIT (số văn bản, rỗng = đủ).
Đầu ra /kaggle/working/out/<bộ>/: <hệ>/canon_kg.txt (+ đầu ra từng bước), <hệ>_eval.txt, scores_exact.json,
support.json (+ .jsonl), redundancy.json; out/summary.json, logs/, server.log.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

DATASET = os.environ.get("EXTRA_DATASET", "nyt11")
SYSTEMS = [s for s in os.environ.get("EXTRA_SYSTEMS", "mine,edc,atom_edc,kggen_edc").split(",") if s]
LIMIT = int(os.environ["EXTRA_LIMIT"]) if os.environ.get("EXTRA_LIMIT") else None
EDC_TIMEOUT_H, OIE_TIMEOUT_H = 5, 5
WORKERS = 4
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
EDC_REPO, EDC_COMMIT = "https://github.com/clear-nus/edc.git", "02aad17e2fc1e9845c84746b384e7fd3160edc1a"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
WORK = "/kaggle/working"
OUT, LOGS = f"{WORK}/out/{DATASET}", f"{WORK}/out/logs"
KLTN, EDC = f"{WORK}/kltn", f"{WORK}/edc"
SERVER = "http://127.0.0.1:8080"
T0 = time.time()


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


def find_data(name: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if os.path.basename(dirpath) == name and "input.txt" in filenames:
            return dirpath
    return None


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


for d in (OUT, LOGS):
    os.makedirs(d, exist_ok=True)
summary: dict = {"dataset": DATASET, "systems": SYSTEMS, "limit": LIMIT, "runs": {}, "notes": []}

# 1. GPU, code, gói, dữ liệu
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
if not os.path.exists(KLTN):
    sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser nervaluate==0.1.8 unidecode beautifulsoup4 regex 'openai>=1'")
sh(f"git clone -q {EDC_REPO} {EDC} && git -C {EDC} checkout -q {EDC_COMMIT} && python {KLTN}/scripts/edc_patch.py {EDC}")
src_dir = find_data(DATASET)
assert src_dir, f"không thấy thư mục dữ liệu {DATASET} trong /kaggle/input"
DATA = {k: f"{src_dir}/{f}" for k, f in (("schema", "schema.csv"), ("oie_fs", "oie_few_shot_examples.txt"),
                                         ("sd_fs", "sd_few_shot_examples.txt"))}
lines = open(f"{src_dir}/input.txt", encoding="utf-8").read().splitlines()[:LIMIT]
refs = open(f"{src_dir}/ref.txt", encoding="utf-8").read().splitlines()[:LIMIT]
assert len(lines) == len(refs), (len(lines), len(refs))
DATA["input"], DATA["ref"] = f"{WORK}/{DATASET}_in.txt", f"{WORK}/{DATASET}_ref.txt"
open(DATA["input"], "w", encoding="utf-8").write("\n".join(lines) + "\n")
open(DATA["ref"], "w", encoding="utf-8").write("\n".join(refs) + "\n")
N = summary["n"] = len(lines)


def venv(name: str, pkgs: str) -> str:
    path = f"{WORK}/venv_{name}"
    try:
        sh(f"pip install -q uv && uv venv -q --python 3.11 {path} && uv pip install -q --python {path}/bin/python {pkgs}")
        return f"{path}/bin/python"
    except subprocess.CalledProcessError:
        summary["notes"].append(f"venv {name} lỗi → cài vào python hệ thống")
        sh(f"pip install -q {pkgs}")
        return "python"


py_atom = venv("atom", "itext2kg==1.1.0 dateparser") if "atom_edc" in SYSTEMS else None
py_kggen = venv("kggen", "kg-gen==0.4.0") if "kggen_edc" in SYSTEMS else None

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
    server = subprocess.Popen(cmd, shell=True, stdout=open(f"{WORK}/out/server.log", "ab"), stderr=subprocess.STDOUT,
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
       "KGU_LLM_EXTRA_BODY": "{}", "KGU_LLM_MAX_TOKENS": "4096", "OPENAI_API_KEY": "none"}
two_gpu = len(gpus) >= 2
emb = "intfloat/e5-mistral-7b-instruct" if two_gpu else "intfloat/multilingual-e5-large-instruct"
summary["edc_embedder"] = emb
llm = "gpt-gemma-4-e4b"      # chứa "gpt" → nhánh OpenAI của EDC; llama-server bỏ qua tên model
edc_env = {**env, "OPENAI_KEY": "none", "OPENAI_BASE_URL": f"{SERVER}/v1", "PYTHONPATH": EDC,
           "EDC_EMB_DEVICE": "cuda:1" if two_gpu else "cuda:0"}


def timed(run: str, cmd: str, **kw) -> bool:
    t0 = time.time()
    try:
        sh(f"{cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0", executable="/bin/bash", **kw)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    sec = round(time.time() - t0, 1)
    summary["runs"][run] = {"ok": ok, "seconds": sec, "sec_per_item": round(sec / max(N, 1), 2)}
    print(f"=== {run}: ok={ok} {sec}s, tổng {round((time.time() - T0) / 60)} phút", flush=True)
    return ok


def run_edc(system: str, oie_json: str | None) -> None:
    """EDC đầy đủ (oie_json None) hoặc chỉ Schema Definition + Canonicalization trên triple có sẵn."""
    edc_out = f"{WORK}/edc_run_{system}"
    e = {**edc_env, **({"EDC_OIE_FROM": oie_json} if oie_json else {})}
    if timed(system, f"timeout {EDC_TIMEOUT_H}h python -u run.py --oie_llm {llm} --sd_llm {llm} --sc_llm {llm} "
                     f"--sc_embedder {emb} --oie_few_shot_example_file_path {DATA['oie_fs']} "
                     f"--sd_few_shot_example_file_path {DATA['sd_fs']} --input_text_file_path {DATA['input']} "
                     f"--target_schema_path {DATA['schema']} --output_dir {edc_out} --logging_verbose", cwd=EDC, env=e):
        os.makedirs(f"{OUT}/{system}", exist_ok=True)
        for name in ("oie_kg.txt", "schema_definition.txt", "canon_kg.txt", "result_at_each_stage.json"):
            if os.path.exists(f"{edc_out}/iter0/{name}"):
                shutil.copy(f"{edc_out}/iter0/{name}", f"{OUT}/{system}/{name}")


# 3. các hệ
if "mine" in SYSTEMS:
    os.makedirs(f"{OUT}/mine", exist_ok=True)
    timed("mine", f"python -u {KLTN}/scripts/run_webnlg.py --input {DATA['input']} --schema {DATA['schema']} "
                  f"--examples {DATA['oie_fs']} --out {OUT}/mine/canon_kg.txt --workers {WORKERS}", env=env)
if "edc" in SYSTEMS:
    run_edc("edc", None)
for system, script, py in (("atom_edc", "atom_edc.py", py_atom), ("kggen_edc", "kggen_edc.py", py_kggen)):
    if system in SYSTEMS:
        raw = system.replace("_edc", "")
        os.makedirs(f"{OUT}/{raw}", exist_ok=True)
        if timed(raw, f"timeout {OIE_TIMEOUT_H}h {py} -u -W ignore {KLTN}/scripts/{script} --input {DATA['input']} "
                      f"--out {OUT}/{raw}/oie_kg.txt", env=env):
            run_edc(system, f"{OUT}/{raw}/oie_kg.txt.json")
FINAL = {s: f"{OUT}/{s}/canon_kg.txt" for s in SYSTEMS if os.path.exists(f"{OUT}/{s}/canon_kg.txt")}
RAW = {r: f"{OUT}/{r}/oie_kg.txt" for r in ("atom", "kggen") if os.path.exists(f"{OUT}/{r}/oie_kg.txt")}

# 4. độ chính xác theo văn bản nguồn (judge Gemma, còn server)
if FINAL:
    specs = " ".join(f"--system {DATASET}/{s}=lines:{p}" for s, p in FINAL.items())
    timed("judge", f"python -u {KLTN}/scripts/judge_support.py {specs} --lines-input {DATA['input']} "
                   f"--out {OUT}/support.json", env=env)
server.kill()


# 5. chấm gold + độ gọn
def parse_eval(text: str) -> dict:
    out = {}
    for block in ("Partial", "Strict", "Exact"):
        m = re.search(rf"\n{block}\n.*?Precision: ([\d.]+) Recall: ([\d.]+)\nF1: ([\d.]+)", text, re.S)
        if m:
            out[block] = {k: round(float(v), 4) for k, v in zip(("P", "R", "F1"), m.groups())}
    return out


def n_parseable(path: str) -> dict:
    """Script chấm EDC lặng lẽ bỏ dòng không parse được (cả gold tương ứng) → báo số dòng thực sự được chấm."""
    ok = empty = n_triples = 0
    rows = open(path, encoding="utf-8").read().splitlines()
    for line in rows:
        try:
            t = ast.literal_eval(line.strip())
            assert all(len(x) == 3 and all(isinstance(e, str) for e in x) for x in t)
            ok, empty, n_triples = ok + 1, empty + (not t), n_triples + len(t)
        except Exception:
            pass
    return {"lines": len(rows), "scored": ok, "empty": empty, "avg_triples": round(n_triples / max(ok, 1), 2)}


summary["edc_script"] = {}
for system, cand in {**FINAL, **RAW}.items():
    try:
        text = sh(f"python {KLTN}/scripts/edc_eval_fast.py --edc_script evaluate/evaluation_script.py "
                  f"--edc_output {cand} --reference {DATA['ref']}", cwd=EDC, capture=True, env=env)
        open(f"{OUT}/{system}_eval.txt", "w", encoding="utf-8").write(text)
        summary["edc_script"][system] = {**parse_eval(text), **n_parseable(cand)}
    except subprocess.CalledProcessError as e:
        print(f"!!! chấm {system} lỗi: {e.stdout}\n{e.stderr}", flush=True)
        summary["edc_script"][system] = {"error": True, **n_parseable(cand)}
preds = " ".join(f"--pred {s}={p}" for s, p in {**FINAL, **RAW}.items())
for step, cmd in (("exact", f"python {KLTN}/scripts/score_nyt11.py --ref {DATA['ref']} {preds} "
                            f"--out {OUT}/scores_exact.json"),
                  ("redundancy", f"python {KLTN}/scripts/kg_redundancy.py "
                                 + " ".join(f"--system {DATASET}/{s}=lines:{p}" for s, p in FINAL.items())
                                 + f" --lines-input {DATA['input']} --scope doc --device cuda "
                                   f"--out {OUT}/redundancy.json")):
    if FINAL:
        try:
            sh(f"set -o pipefail; {cmd} 2>&1 | tee {LOGS}/{step}.log", executable="/bin/bash", env=env)
        except subprocess.CalledProcessError:
            print(f"!!! {step} lỗi, xem logs/{step}.log", flush=True)
for name in ("scores_exact", "support", "redundancy"):
    if os.path.exists(f"{OUT}/{name}.json"):
        summary[name] = json.load(open(f"{OUT}/{name}.json"))
for r in RAW:
    if os.path.exists(f"{OUT}/{r}/oie_kg.txt.stats.json"):
        summary["runs"][r]["stats"] = json.load(open(f"{OUT}/{r}/oie_kg.txt.stats.json"))

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{WORK}/out/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False)[:20000], flush=True)
for d in (KLTN, EDC, f"{WORK}/models", f"{WORK}/llama-bin", f"{WORK}/venv_atom", f"{WORK}/venv_kggen",
          *(f"{WORK}/edc_run_{s}" for s in SYSTEMS)):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model/venv vào output
print("DONE", flush=True)
