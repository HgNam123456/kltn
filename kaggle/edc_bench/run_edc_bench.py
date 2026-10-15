"""Kaggle kernel: so pipeline của mình với EDC gốc trên hai benchmark còn lại của paper EDC (Zhang & Soh, EMNLP 2024):
REBEL và Wiki-NRE (bản trong repo clear-nus/edc ghim EDC_COMMIT; mỗi bộ 1.000 câu, schema 196 / 45 quan hệ). Cùng cách
làm với WebNLG ở kernel kgu-home-benchmarks (kaggle/home_bench/run_home_bench.py): mọi hệ thống dùng Gemma 4 E4B qua
llama-server CUDA. EDC: OIE → schema definition → schema canonicalization theo target schema (không refinement),
embedder e5-mistral-7b-instruct fp16 trên GPU thứ hai. Mình: scripts/run_webnlg.py (một lệnh gọi/câu, quan hệ ép theo
schema của bộ, cùng file few-shot OIE của EDC cho bộ đó). Chấm evaluate/evaluation_script.py của EDC
(Partial/Strict/Exact, không LLM judge) qua scripts/edc_eval_fast.py: script gốc thử mọi hoán vị ghép triple (n! với
n = max(số dự đoán, số gold); lượt chạy thử: 30 câu REBEL của EDC mất 10,9 phút, câu 12+ triple không bao giờ xong) →
bản nhanh chỉ thay bước tìm hoán vị bằng gán tối ưu chọn đúng hoán vị gốc chọn (kiểm toàn văn: kgu-edc-eval-check).

Prompt REBEL của mình ~28k ký tự (196 quan hệ kèm mô tả + enum trong JSON schema) → mỗi slot CTX_PER_SLOT token thay vì
8.192, để prompt + max_tokens 4.096 vẫn vừa ngữ cảnh.
Vá EDC giống kgu-home-benchmarks: tên model chứa "gpt" (EDC đi nhánh OpenAI) + OPENAI_BASE_URL; embedder nạp fp16
(bản gốc fp32 = 28 GB); import e5_mistral_utils (chỉ cho refinement, nạp huggingface_hub.Repository đã bị gỡ) bọc
try/except; lỗi gốc ở nhánh OpenAI (giải phóng verify model chưa từng nạp) bọc điều kiện.

Code lấy từ repo kltn nhánh KGU_BRANCH. Dataset ngocnam2005/llama-server-cuda (binary đã build).
LIMIT = số câu mỗi bộ (None = đủ). DATASETS chọn bộ, RUNS chọn hệ thống.
Đầu ra /kaggle/working/out/: <bộ>/<hệ>/canon_kg.txt, <bộ>/<hệ>_eval.txt, summary.json, logs/*.log, server.log.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

LIMIT = None
DATASETS = ["rebel", "wiki-nre"]
RUNS = ["mine", "edc"]
EDC_TIMEOUT_H = 4                     # mỗi bộ; chặn EDC (gọi LLM tuần tự, lỗi API thì thử lại vô hạn)
WORKERS = 4
CTX_PER_SLOT = 12288
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
EDC_REPO, EDC_COMMIT = "https://github.com/clear-nus/edc.git", "02aad17e2fc1e9845c84746b384e7fd3160edc1a"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
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


for d in (OUT, LOGS, *(f"{OUT}/{ds}" for ds in DATASETS)):
    os.makedirs(d, exist_ok=True)
summary: dict = {"limit": LIMIT, "datasets": DATASETS, "runs": {}, "notes": []}

# 1. GPU, code, gói
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
print(gpus, flush=True)
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q nervaluate==0.1.8 unidecode beautifulsoup4 regex 'openai>=1'")  # script EDC cần API nervaluate 0.1
sh(f"git clone -q {EDC_REPO} {EDC} && git -C {EDC} checkout -q {EDC_COMMIT}")

# 2. dữ liệu (cắt LIMIT): input và gold phải khớp từng dòng
DATA: dict[str, dict] = {}
for ds in DATASETS:
    lines = open(f"{EDC}/datasets/{ds}.txt", encoding="utf-8").read().splitlines()[:LIMIT]
    refs = open(f"{EDC}/evaluate/references/{ds}.txt", encoding="utf-8").read().splitlines()[:LIMIT]
    assert len(lines) == len(refs), (ds, len(lines), len(refs))
    DATA[ds] = {"input": f"{WORK}/{ds}_in.txt", "ref": f"{WORK}/{ds}_ref.txt", "n": len(lines)}
    open(DATA[ds]["input"], "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(DATA[ds]["ref"], "w", encoding="utf-8").write("\n".join(refs) + "\n")
summary["n"] = {ds: DATA[ds]["n"] for ds in DATASETS}

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
    COMMON + f" --parallel {WORKERS} --cont-batching --kv-unified-per-slot {ctx} --reasoning off --flash-attn on"
             f" --spec-type draft-mtp --spec-draft-n-max 2 --model-draft {draft}"
    for ctx in (CTX_PER_SLOT, 8192)
] + [COMMON + f" --parallel {WORKERS} --cont-batching -c {8192 * WORKERS} --flash-attn on"]


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
       "KGU_LLM_EXTRA_BODY": "{}", "KGU_LLM_MAX_TOKENS": "4096", "OPENAI_API_KEY": "none"}


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


# 4. vá EDC (một lần cho mọi bộ)
two_gpu = len(gpus) >= 2
if "edc" in RUNS:
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
    patched = [f for f in sorted(os.listdir(f"{EDC}/edc")) if f.endswith(".py")
               and imp in open(f"{EDC}/edc/{f}", encoding="utf-8").read()]
    assert "edc_framework.py" in patched, "vá import EDC không khớp source"
    for f in patched:
        src = open(f"{EDC}/edc/{f}", encoding="utf-8").read()
        open(f"{EDC}/edc/{f}", "w", encoding="utf-8").write(src.replace(imp, (
            f"try:\n    {imp}\nexcept Exception:  # chỉ dùng cho refinement có adapter\n"
            "    MistralForSequenceEmbedding = None")))
    src = open(fw, encoding="utf-8").read()        # lỗi gốc: nhánh OpenAI vẫn giải phóng verify model chưa nạp
    old = ("            llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
           "            del self.loaded_model_dict[self.sc_llm_name]\n")
    assert src.count(old) == 1, "vá free_model EDC không khớp source"
    open(fw, "w", encoding="utf-8").write(src.replace(old, (
        "            if not llm_utils.is_model_openai(self.sc_llm_name):\n"
        "                llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
        "                del self.loaded_model_dict[self.sc_llm_name]\n")))
    edc_env = {**env, "OPENAI_KEY": "none", "OPENAI_BASE_URL": f"{SERVER}/v1", "PYTHONPATH": EDC,
               "EDC_EMB_DEVICE": "cuda:1" if two_gpu else "cuda:0"}
    llm = "gpt-gemma-4-e4b"      # chứa "gpt" → nhánh OpenAI của EDC; llama-server bỏ qua tên model

# 5. hệ thống, từng bộ
for ds in DATASETS:
    few_oie = f"{EDC}/few_shot_examples/{ds}/oie_few_shot_examples.txt"
    n = DATA[ds]["n"]
    if "mine" in RUNS:
        os.makedirs(f"{OUT}/{ds}/mine", exist_ok=True)
        timed(f"mine_{ds}", f"python -u {KLTN}/scripts/run_webnlg.py --input {DATA[ds]['input']} "
                            f"--schema {EDC}/schemas/{ds}_schema.csv --examples {few_oie} "
                            f"--out {OUT}/{ds}/mine/canon_kg.txt --workers {WORKERS}", n, env=env)
    if "edc" in RUNS:
        edc_out = f"{WORK}/edc_run_{ds}"
        if timed(f"edc_{ds}", f"timeout {EDC_TIMEOUT_H}h python -u run.py --oie_llm {llm} --sd_llm {llm} "
                              f"--sc_llm {llm} --sc_embedder {emb} --oie_few_shot_example_file_path {few_oie} "
                              f"--sd_few_shot_example_file_path ./few_shot_examples/{ds}/sd_few_shot_examples.txt "
                              f"--input_text_file_path {DATA[ds]['input']} "
                              f"--target_schema_path ./schemas/{ds}_schema.csv "
                              f"--output_dir {edc_out} --logging_verbose", n, cwd=EDC, env=edc_env):
            os.makedirs(f"{OUT}/{ds}/edc", exist_ok=True)
            for name in ("oie_kg.txt", "schema_definition.txt", "canon_kg.txt"):
                if os.path.exists(f"{edc_out}/iter0/{name}"):
                    shutil.copy(f"{edc_out}/iter0/{name}", f"{OUT}/{ds}/edc/{name}")
server.kill()


# 6. chấm
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
    lines_ = open(path, encoding="utf-8").read().splitlines()
    for line in lines_:
        try:
            t = ast.literal_eval(line.strip())
            assert all(len(x) == 3 and all(isinstance(e, str) for e in x) for x in t)
            ok += 1
            empty += not t
            n_triples += len(t)
        except Exception:
            pass
    return {"lines": len(lines_), "scored": ok, "empty": empty, "avg_triples": round(n_triples / max(ok, 1), 2)}


for ds in DATASETS:
    summary[ds] = {}
    for system in ("edc", "mine"):
        cand = f"{OUT}/{ds}/{system}/canon_kg.txt"
        if not os.path.exists(cand):
            continue
        t0 = time.time()
        try:
            text = sh(f"python {KLTN}/scripts/edc_eval_fast.py --edc_script evaluate/evaluation_script.py "
                      f"--edc_output {cand} --reference {DATA[ds]['ref']}", cwd=EDC, capture=True, env=env)
            open(f"{OUT}/{ds}/{system}_eval.txt", "w", encoding="utf-8").write(text)
            summary[ds][system] = {**parse_eval(text), **n_parseable(cand)}
        except subprocess.CalledProcessError as e:
            print(f"!!! chấm {ds} {system} lỗi: {e.stdout}\n{e.stderr}", flush=True)
            summary[ds][system] = {"error": True, **n_parseable(cand)}
        summary[ds][system]["score_minutes"] = round((time.time() - t0) / 60, 1)
        print(f"=== chấm {ds} {system}: {summary[ds][system]}", flush=True)

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)

for d in (KLTN, EDC, f"{WORK}/models", f"{WORK}/llama-bin", *(f"{WORK}/edc_run_{ds}" for ds in DATASETS)):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model vào output
print("DONE", flush=True)
