"""Kaggle kernel: so pipeline của mình với baseline trên benchmark sân nhà của chính baseline, mọi hệ thống cùng
Gemma 4 E4B qua llama-server CUDA:
- WebNLG test của EDC (Zhang & Soh, EMNLP 2024; 1.165 câu, schema 159 quan hệ): EDC gốc clear-nus/edc ghim
  EDC_COMMIT (OIE → schema definition → schema canonicalization, không refinement) vs mình (scripts/run_webnlg.py).
  Chấm evaluate/evaluation_script.py của EDC (Partial/Strict/Exact, không LLM judge).
- 2020-COVID-NYT (ATOM, Lairgi et al., EACL 2026; 122 nhóm ngày, 3.918 gold 5-tuple): ATOM 2 bước, ATOM trích thẳng
  (itext2kg 1.1.0, scripts/atom_covid.py), KGGen 0.4.0 (scripts/kggen_covid.py) vs mình (scripts/run_covid_nyt.py).
  Chấm scripts/score_covid_nyt.py với all-mpnet-base-v2 @ 0,7 (hiệu chỉnh: results/kaggle/covid_calibration.json),
  kèm --calibrate để số các cột GPT-4.1 phát hành nằm cùng bảng.
Vá EDC để chạy với llama-server: tên model chứa "gpt" (EDC đi nhánh OpenAI) + OPENAI_BASE_URL; embedder nạp fp16 trên
GPU thứ hai (bản gốc nạp fp32 = 28 GB). Chỉ có 1 GPU → intfloat/multilingual-e5-large-instruct (ghi trong summary).
Vá import của EDC: edc/utils/e5_mistral_utils.py (chỉ dùng cho refinement có adapter) nạp huggingface_hub.Repository và
transformers.utils.get_full_repo_name đã bị gỡ ở bản mới → bọc try/except.
itext2kg 1.1.0 (ghim numpy<2) cài trong venv sạch; kg-gen 0.4.0 cài vào python hệ thống sau khi EDC chạy xong (như
kernel baselines).

Code + data COVID lấy từ repo kltn nhánh KGU_BRANCH. Dataset ngocnam2005/llama-server-cuda (binary đã build).
LIMIT = số câu WebNLG / số nhóm ngày COVID (None = đủ). BENCH chọn benchmark, RUNS chọn hệ thống.
Đầu ra /kaggle/working/out/: webnlg/<hệ>/canon_kg.txt + <hệ>_eval.txt, covid/<hệ>.jsonl, covid/scores.json,
summary.json, logs/*.log, server.log.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

LIMIT = 20
BENCH = ["webnlg", "covid"]
RUNS = ["mine_webnlg", "edc", "mine_covid", "atom_direct", "atom", "kggen"]
EDC_TIMEOUT_H = 6                     # chặn EDC (gọi LLM tuần tự, lỗi API thì thử lại vô hạn)
WORKERS = 4
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
EDC_REPO, EDC_COMMIT = "https://github.com/clear-nus/edc.git", "02aad17e2fc1e9845c84746b384e7fd3160edc1a"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
SCORE_EMBEDDER, SCORE_THRESHOLD = "sentence-transformers/all-mpnet-base-v2", "0.7"
WORK = "/kaggle/working"
OUT, LOGS = f"{WORK}/out", f"{WORK}/out/logs"
KLTN, EDC = f"{WORK}/kltn", f"{WORK}/edc"
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


for d in (OUT, LOGS, f"{OUT}/webnlg", f"{OUT}/covid"):
    os.makedirs(d, exist_ok=True)
summary: dict = {"limit": LIMIT, "bench": BENCH, "runs": {}, "notes": []}

# 1. GPU, code, gói
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
print(gpus, flush=True)
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser nervaluate unidecode beautifulsoup4 regex 'openai>=1'")


def venv(name: str, pkgs: str) -> str:
    """venv sạch (itext2kg ghim numpy<2, không trộn với gói hệ thống); lỗi tạo venv → dùng python hệ thống."""
    path = f"{WORK}/venv_{name}"
    try:
        sh(f"python -m venv {path} && {path}/bin/pip install -q {pkgs}")
        return f"{path}/bin/python"
    except subprocess.CalledProcessError:
        summary["notes"].append(f"venv {name} lỗi → cài vào python hệ thống")
        sh(f"pip install -q {pkgs}")
        return "python"


# 2. dữ liệu (cắt LIMIT)
if "webnlg" in BENCH:
    sh(f"git clone -q {EDC_REPO} {EDC} && git -C {EDC} checkout -q {EDC_COMMIT}")
    lines = open(f"{EDC}/datasets/webnlg.txt", encoding="utf-8").read().splitlines()[:LIMIT]
    refs = open(f"{EDC}/evaluate/references/webnlg.txt", encoding="utf-8").read().splitlines()[:LIMIT]
    assert len(lines) == len(refs), (len(lines), len(refs))
    WEBNLG_IN, WEBNLG_REF = f"{WORK}/webnlg_in.txt", f"{WORK}/webnlg_ref.txt"
    open(WEBNLG_IN, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(WEBNLG_REF, "w", encoding="utf-8").write("\n".join(refs) + "\n")
    summary["webnlg_n"] = len(lines)
if "covid" in BENCH:
    rows = sorted(read_jsonl(f"{KLTN}/data/covid_nyt/2020_covid_nyt.jsonl"), key=lambda r: (r["date"], r["row"]))
    rows = rows[:LIMIT]
    COVID = f"{WORK}/covid.jsonl"
    with open(COVID, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary["covid_n"] = len(rows)

# 3. llama-server (giống kaggle/baselines/run_baselines.py), ghim GPU 0
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
       "KGU_LLM_EXTRA_BODY": "{}", "OPENAI_API_KEY": "none"}


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


# 4. hệ thống
if "webnlg" in BENCH:
    few_oie = f"{EDC}/few_shot_examples/webnlg/oie_few_shot_examples.txt"
    if "mine_webnlg" in RUNS:
        os.makedirs(f"{OUT}/webnlg/mine", exist_ok=True)
        timed("mine_webnlg", f"python -u {KLTN}/scripts/run_webnlg.py --input {WEBNLG_IN} "
                             f"--schema {EDC}/schemas/webnlg_schema.csv --examples {few_oie} "
                             f"--out {OUT}/webnlg/mine/canon_kg.txt --workers {WORKERS}", len(lines), env=env)
    if "edc" in RUNS:
        two_gpu = len(gpus) >= 2
        emb = "intfloat/e5-mistral-7b-instruct" if two_gpu else "intfloat/multilingual-e5-large-instruct"
        summary["edc_embedder"] = emb
        if not two_gpu:
            summary["notes"].append("1 GPU: EDC dùng multilingual-e5-large-instruct thay e5-mistral-7b-instruct")
        fw = f"{EDC}/edc/edc_framework.py"
        src = open(fw, encoding="utf-8").read()
        old = "model = SentenceTransformer(model_name, trust_remote_code=True)"
        assert src.count(old) == 1, "vá EDC không khớp source"
        open(fw, "w", encoding="utf-8").write(src.replace(old, (
            "model = SentenceTransformer(model_name, trust_remote_code=True, device=os.environ['EDC_EMB_DEVICE'], "
            "model_kwargs={'torch_dtype': __import__('torch').float16})")))
        imp = "from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding"
        for f in ("edc_framework.py", "schema_retriever.py", "entity_extraction.py"):
            src = open(f"{EDC}/edc/{f}", encoding="utf-8").read()
            assert imp in src, f"vá import EDC không khớp {f}"
            open(f"{EDC}/edc/{f}", "w", encoding="utf-8").write(src.replace(imp, (
                f"try:\n    {imp}\nexcept Exception:  # chỉ dùng cho refinement có adapter\n"
                "    MistralForSequenceEmbedding = None")))
        edc_env = {**env, "OPENAI_KEY": "none", "OPENAI_BASE_URL": f"{SERVER}/v1", "PYTHONPATH": EDC,
                   "EDC_EMB_DEVICE": "cuda:1" if two_gpu else "cuda:0"}
        llm = "gpt-gemma-4-e4b"      # chứa "gpt" → nhánh OpenAI của EDC; llama-server bỏ qua tên model
        edc_out = f"{WORK}/edc_run"
        if timed("edc", f"timeout {EDC_TIMEOUT_H}h python -u run.py --oie_llm {llm} --sd_llm {llm} --sc_llm {llm} "
                        f"--sc_embedder {emb} --oie_few_shot_example_file_path {few_oie} "
                        f"--sd_few_shot_example_file_path ./few_shot_examples/webnlg/sd_few_shot_examples.txt "
                        f"--input_text_file_path {WEBNLG_IN} --target_schema_path ./schemas/webnlg_schema.csv "
                        f"--output_dir {edc_out} --logging_verbose", len(lines), cwd=EDC, env=edc_env):
            os.makedirs(f"{OUT}/webnlg/edc", exist_ok=True)
            shutil.copy(f"{edc_out}/iter0/canon_kg.txt", f"{OUT}/webnlg/edc/canon_kg.txt")
            for name in ("oie_kg.txt", "schema_definition.txt", "canon_kg.txt"):
                if os.path.exists(f"{edc_out}/iter0/{name}"):
                    shutil.copy(f"{edc_out}/iter0/{name}", f"{OUT}/webnlg/edc/{name}")

if "covid" in BENCH:
    n_covid = len(rows)
    if "mine_covid" in RUNS:
        timed("mine_covid", f"python -u {KLTN}/scripts/run_covid_nyt.py --data {COVID} --out {OUT}/covid/mine.jsonl",
              n_covid, env=env)
    if {"atom", "atom_direct"} & set(RUNS):
        py_atom = venv("atom", "itext2kg==1.1.0 dateparser")
        for run, mode in (("atom_direct", "direct"), ("atom", "atom")):
            if run in RUNS:
                timed(run, f"{py_atom} -u -W ignore {KLTN}/scripts/atom_covid.py --mode {mode} --data {COVID} "
                           f"--out {OUT}/covid/{run.replace('_', '-')}.jsonl", n_covid, env=env)
    if "kggen" in RUNS:
        sh("pip install -q kg-gen==0.4.0")
        timed("kggen", f"python -u {KLTN}/scripts/kggen_covid.py --data {COVID} --out {OUT}/covid/kggen.jsonl",
              n_covid, env=env)
server.kill()


# 5. chấm
def parse_webnlg_eval(text: str) -> dict:
    out = {}
    for block in ("Partial", "Strict", "Exact"):
        m = re.search(rf"\n{block}\n.*?Precision: ([\d.]+) Recall: ([\d.]+)\nF1: ([\d.]+)", text, re.S)
        if m:
            out[block] = {k: round(float(v), 4) for k, v in zip(("P", "R", "F1"), m.groups())}
    return out


def n_parseable(path: str) -> dict:
    """Script chấm EDC lặng lẽ bỏ dòng không parse được (cả gold tương ứng) → báo số dòng thực sự được chấm."""
    ok = empty = 0
    lines_ = open(path, encoding="utf-8").read().splitlines()
    for line in lines_:
        try:
            t = ast.literal_eval(line.strip())
            assert all(len(x) == 3 and all(isinstance(e, str) for e in x) for x in t)
            ok += 1
            empty += not t
        except Exception:
            pass
    return {"lines": len(lines_), "scored": ok, "empty": empty}


if "webnlg" in BENCH:
    summary["webnlg"] = {}
    for system in ("edc", "mine"):
        cand = f"{OUT}/webnlg/{system}/canon_kg.txt"
        if not os.path.exists(cand):
            continue
        try:
            text = sh(f"python evaluate/evaluation_script.py --edc_output {cand} --reference {WEBNLG_REF}",
                      cwd=EDC, capture=True, env=env)
            open(f"{OUT}/webnlg/{system}_eval.txt", "w", encoding="utf-8").write(text)
            summary["webnlg"][system] = {**parse_webnlg_eval(text), **n_parseable(cand)}
        except subprocess.CalledProcessError as e:
            print(f"!!! chấm webnlg {system} lỗi: {e.stdout}\n{e.stderr}", flush=True)
            summary["webnlg"][system] = {"error": True, **n_parseable(cand)}

if "covid" in BENCH:
    preds = [f"--pred {p[:-6]}={OUT}/covid/{p}" for p in sorted(os.listdir(f"{OUT}/covid")) if p.endswith(".jsonl")]
    try:
        sh(f"python -u {KLTN}/scripts/score_covid_nyt.py --data {COVID} --calibrate {' '.join(preds)} "
           f"--embedder {SCORE_EMBEDDER} --threshold {SCORE_THRESHOLD} --device cuda "
           f"--out {OUT}/covid/scores.json 2>&1 | tee {LOGS}/score_covid.log", executable="/bin/bash", env=env)
        summary["covid"] = json.load(open(f"{OUT}/covid/scores.json"))["results"][SCORE_EMBEDDER][SCORE_THRESHOLD]
    except subprocess.CalledProcessError:
        print("!!! chấm covid lỗi, xem logs/score_covid.log", flush=True)
    summary["covid_outputs"] = {}
    for p in sorted(os.listdir(f"{OUT}/covid")):
        if p.endswith(".jsonl"):
            rs = read_jsonl(f"{OUT}/covid/{p}")
            summary["covid_outputs"][p[:-6]] = {
                "rows": len(rs), "empty": sum(not r["quintuples"] for r in rs),
                "avg_quintuples": round(sum(len(r["quintuples"]) for r in rs) / max(len(rs), 1), 2),
                "with_time": round(sum(bool(q[3] or q[4]) for r in rs for q in r["quintuples"])
                                   / max(sum(len(r["quintuples"]) for r in rs), 1), 3)}

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)

for d in (KLTN, EDC, f"{WORK}/models", f"{WORK}/llama-bin", f"{WORK}/venv_atom",
          f"{WORK}/edc_run", f"{EDC}/result_xmls"):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model/venv vào output
print("DONE", flush=True)
