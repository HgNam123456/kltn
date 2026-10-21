"""Runner chung cho kernel kgu-atom-edc-*: ATOM (Lairgi et al., Findings EACL 2026; itext2kg 1.1.0) trên ba benchmark
của EDC (WebNLG / REBEL / Wiki-NRE trong repo clear-nus/edc ghim EDC_COMMIT), Gemma 4 E4B qua llama-server CUDA như
kgu-home-benchmarks và kgu-edc-rebel-wikinre. Hai hệ được chấm:
- atom: đầu ra thô của ATOM (scripts/atom_edc.py: factoid rồi 5-tuple theo từng factoid, prompt của ATOM; bỏ thời gian)
  → quan hệ tự do, không thuộc schema của bộ; chỉ để tham khảo.
- atom_edc: triple thô của ATOM thay cho bước OIE của EDC, rồi Schema Definition + Schema Canonicalization của EDC theo
  target schema (không refinement), embedder e5-mistral-7b-instruct fp16 trên GPU thứ hai — đúng phần sau OIE của EDC
  ở kgu-edc-rebel-wikinre / kgu-home-benchmarks, nên so với dòng EDC ở đó chỉ khác bước trích.
- atom_time_edc: như atom_edc nhưng thêm một triple ngày cho mỗi t_start/t_end của ATOM (scripts/atom_edc.py, file
  .time.json), vì ATOM không đặt ngày làm đuôi triple mà gold của EDC có quan hệ ngày; chỉ chạy khi số triple thêm vào
  ≥ TIME_MIN_SHARE số triple (Wiki-NRE gần như không có ngày).
Vá EDC như các kernel trước (tên model chứa "gpt" + OPENAI_BASE_URL; embedder fp16; import e5_mistral_utils bọc
try/except; free_model của nhánh OpenAI bọc điều kiện) và thêm: EDC.oie đọc triple có sẵn từ $EDC_OIE_FROM.
Chấm evaluate/evaluation_script.py của EDC qua scripts/edc_eval_fast.py.

Kernel gọi file này qua stub (kaggle/atom_edc/<tên>/kernel.py) đặt ATOM_EDC_DATASETS (vd. "webnlg" hoặc
"rebel,wiki-nre") và ATOM_EDC_LIMIT (số câu mỗi bộ, rỗng = đủ).
Đầu ra /kaggle/working/out/: <bộ>/atom/oie_kg.txt(+.json, .factoids.json, .stats.json), <bộ>/atom_edc/ và <bộ>/atom_time_edc/
{canon_kg.txt, schema_definition.txt, result_at_each_stage.json}, <bộ>/<hệ>_eval.txt, summary.json, logs/*.log, server.log.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

DATASETS = [d for d in os.environ.get("ATOM_EDC_DATASETS", "webnlg,rebel,wiki-nre").split(",") if d]
LIMIT = int(os.environ["ATOM_EDC_LIMIT"]) if os.environ.get("ATOM_EDC_LIMIT") else None
EDC_TIMEOUT_H = 4                     # mỗi bộ; EDC gọi LLM tuần tự, lỗi API thì thử lại vô hạn
ATOM_TIMEOUT_H = 5
TIME_MIN_SHARE = 0.02
WORKERS = 4
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
if not os.path.exists(KLTN):
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

# 3. llama-server, ghim GPU 0
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
       "KGU_LLM_BASE_URL": f"{SERVER}/v1", "KGU_LLM_API_KEY": "none", "OPENAI_API_KEY": "none"}


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


# 4. ATOM (venv sạch Python 3.11: itext2kg ghim numpy<2, như kgu-home-benchmarks)
venv_atom = f"{WORK}/venv_atom"
try:
    sh(f"pip install -q uv && uv venv -q --python 3.11 {venv_atom} && "
       f"uv pip install -q --python {venv_atom}/bin/python itext2kg==1.1.0")
    py_atom = f"{venv_atom}/bin/python"
except subprocess.CalledProcessError:
    summary["notes"].append("venv atom lỗi → cài itext2kg vào python hệ thống")
    sh("pip install -q itext2kg==1.1.0")
    py_atom = "python"
for ds in DATASETS:
    os.makedirs(f"{OUT}/{ds}/atom", exist_ok=True)
    if timed(f"atom_{ds}", f"timeout {ATOM_TIMEOUT_H}h {py_atom} -u -W ignore {KLTN}/scripts/atom_edc.py "
                           f"--input {DATA[ds]['input']} --out {OUT}/{ds}/atom/oie_kg.txt", DATA[ds]["n"], env=env):
        summary["runs"][f"atom_{ds}"]["stats"] = json.load(open(f"{OUT}/{ds}/atom/oie_kg.txt.stats.json"))

# 5. EDC sau OIE trên triple của ATOM
two_gpu = len(gpus) >= 2
emb = "intfloat/e5-mistral-7b-instruct" if two_gpu else "intfloat/multilingual-e5-large-instruct"
summary["edc_embedder"] = emb
if not two_gpu:
    summary["notes"].append("1 GPU: EDC dùng multilingual-e5-large-instruct thay e5-mistral-7b-instruct")
fw = f"{EDC}/edc/edc_framework.py"
src = open(fw, encoding="utf-8").read()
old = "model = SentenceTransformer(model_name, trust_remote_code=True)"
assert src.count(old) == 1, "vá EDC không khớp source"
src = src.replace(old, ("model = SentenceTransformer(model_name, trust_remote_code=True, device=os.environ['EDC_EMB_DEVICE'], "
                        "model_kwargs={'torch_dtype': __import__('torch').float16})"))
old = ("            llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
       "            del self.loaded_model_dict[self.sc_llm_name]\n")
assert src.count(old) == 1, "vá free_model EDC không khớp source"      # lỗi gốc: nhánh OpenAI giải phóng model chưa nạp
src = src.replace(old, ("            if not llm_utils.is_model_openai(self.sc_llm_name):\n"
                        "                llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
                        "                del self.loaded_model_dict[self.sc_llm_name]\n"))
old = "previous_extracted_triplets_list: List[List[str]] = None, free_model=False\n    ):\n"
assert src.count(old) == 1, "vá EDC.oie không khớp source"
src = src.replace(old, old + (
    "        if os.environ.get('EDC_OIE_FROM'):  # vá kgu: triple OIE có sẵn (ATOM) thay bước OIE của EDC\n"
    "            oie_triples_list = json.load(open(os.environ['EDC_OIE_FROM'], encoding='utf-8'))\n"
    "            assert len(oie_triples_list) == len(input_text_list), (len(oie_triples_list), len(input_text_list))\n"
    "            return oie_triples_list, ['' for _ in input_text_list], ['' for _ in input_text_list]\n"))
open(fw, "w", encoding="utf-8").write(src)
imp = "from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding"
for f in sorted(os.listdir(f"{EDC}/edc")):
    if f.endswith(".py") and imp in (s := open(f"{EDC}/edc/{f}", encoding="utf-8").read()):
        open(f"{EDC}/edc/{f}", "w", encoding="utf-8").write(s.replace(imp, (
            f"try:\n    {imp}\nexcept Exception:  # chỉ dùng cho refinement có adapter\n"
            "    MistralForSequenceEmbedding = None")))
llm = "gpt-gemma-4-e4b"      # chứa "gpt" → nhánh OpenAI của EDC; llama-server bỏ qua tên model
CANON = [("atom_edc", "oie_kg.txt.json"), ("atom_time_edc", "oie_kg.txt.time.json")]
for ds, (system, oie_file) in ((ds, c) for ds in DATASETS for c in CANON):
    oie_json = f"{OUT}/{ds}/atom/{oie_file}"
    if not os.path.exists(oie_json):
        continue
    stats = summary["runs"].get(f"atom_{ds}", {}).get("stats", {})
    if system == "atom_time_edc" and stats.get("time_triples", 0) < TIME_MIN_SHARE * max(stats.get("triples", 0), 1):
        summary["notes"].append(f"{ds}: bỏ atom_time_edc, triple ngày thêm vào {stats.get('time_triples')} "
                                f"/ {stats.get('triples')} triple")
        continue
    edc_out = f"{WORK}/edc_run_{ds}_{system}"
    edc_env = {**env, "OPENAI_KEY": "none", "OPENAI_BASE_URL": f"{SERVER}/v1", "PYTHONPATH": EDC,
               "EDC_EMB_DEVICE": "cuda:1" if two_gpu else "cuda:0", "EDC_OIE_FROM": oie_json}
    if timed(f"{system}_{ds}", f"timeout {EDC_TIMEOUT_H}h python -u run.py --oie_llm {llm} --sd_llm {llm} "
                               f"--sc_llm {llm} --sc_embedder {emb} "
                               f"--oie_few_shot_example_file_path ./few_shot_examples/{ds}/oie_few_shot_examples.txt "
                               f"--sd_few_shot_example_file_path ./few_shot_examples/{ds}/sd_few_shot_examples.txt "
                               f"--input_text_file_path {DATA[ds]['input']} --target_schema_path ./schemas/{ds}_schema.csv "
                               f"--output_dir {edc_out} --logging_verbose", DATA[ds]["n"], cwd=EDC, env=edc_env):
        os.makedirs(f"{OUT}/{ds}/{system}", exist_ok=True)
        for name in ("schema_definition.txt", "canon_kg.txt", "result_at_each_stage.json"):
            if os.path.exists(f"{edc_out}/iter0/{name}"):
                shutil.copy(f"{edc_out}/iter0/{name}", f"{OUT}/{ds}/{system}/{name}")
server.kill()


# 6. chấm (như kaggle/edc_bench/run_edc_bench.py)
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
    for system, cand in (("atom", f"{OUT}/{ds}/atom/oie_kg.txt"), *((s, f"{OUT}/{ds}/{s}/canon_kg.txt") for s, _ in CANON)):
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

for d in (KLTN, EDC, venv_atom, f"{WORK}/models", f"{WORK}/llama-bin", *(f"{WORK}/edc_run_{ds}_{s}" for ds in DATASETS for s, _ in CANON)):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model/venv vào output
print("DONE", flush=True)
