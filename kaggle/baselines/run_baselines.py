"""Kaggle kernel: chạy lại baseline EMERGE (EDC+ ICL, EDC+ ZS, KGGen) và KARMA bằng Gemma 4 E4B qua llama-server CUDA
trên dev350, rồi gộp prediction vào evaluation_set để kernel emerge-eval-dev350 chấm cạnh bản GPT-5.1.

Đầu vào: dataset ngocnam2005/emerge-dev350-mine (evaluation_set dev350 kèm prediction 13 baseline + mine);
         ngocnam2005/llama-server-cuda (binary đã build). Code EMERGE ghim EMERGE_COMMIT, KARMA ghim KARMA_COMMIT
         (bản gốc 03/2025), wrapper KARMA lấy từ repo kltn (scripts/karma_emerge.py).
Đầu ra: /kaggle/working/evaluation_set/ (dev350 + prediction mới), raw/ (output thô từng baseline),
        summary.json (giây/instance, số triple), server.log, logs/*.log.
LIMIT = số instance mỗi file delta (None = đủ 10/file = 350 bài). Sửa RUNS để chọn baseline.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

LIMIT = None
RUNS = ["karma_biomed", "karma_general"]    # edc_icl, edc_zs, kggen đã chạy ở v2 (đã có trong dataset)
NAMES = {                                   # tên chuẩn ghi vào predictions, đặt theo kiểu tên của EMERGE
    "edc_icl": "edc-plus-llamacpp/gemma-4-e4b",
    "edc_zs": "edc-plus-zshot-llamacpp/gemma-4-e4b",
    "kggen": "kg-gen/llamacpp/gemma-4-e4b",
    "karma_biomed": "karma-biomed/llamacpp/gemma-4-e4b",     # prompt y sinh nguyên văn
    "karma_general": "karma/llamacpp/gemma-4-e4b",           # chỉ thay cụm chỉ miền (GENERAL_PATCH)
}
EDC_CONFIGS = {"edc_icl": "edc_plus_icl_gpt_5_1", "edc_zs": "edc_plus_zs_gpt_5_1"}   # prompt/gen profile giữ nguyên
WORKERS = 4
EMERGE_REPO, EMERGE_COMMIT = "https://github.com/klimzaporojets/emerge.git", "f68891017e58dfe0d88ef84c19b455dcd8ecaece"
KARMA_REPO, KARMA_COMMIT = "https://github.com/YuxingLu613/KARMA.git", "82d68ba"
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
WORK = "/kaggle/working"
EMERGE = f"{WORK}/emerge"
INPUT = f"{WORK}/input/evaluation_set"
RAW = f"{WORK}/raw"
LOGS = f"{WORK}/logs"
SERVER = "http://127.0.0.1:8080"


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


def find_dataset() -> str:
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if any(d.startswith("snapshot_") for d in dirnames):
            return dirpath
    raise SystemExit("không thấy evaluation_set (snapshot_*/) trong /kaggle/input")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith(".jsonl"):
                yield os.path.join(dirpath, fname)


os.makedirs(RAW, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# 1. input: dev350 (cắt LIMIT bài đầu mỗi file delta khi chạy thử)
src = find_dataset()
for path in jsonl_files(src):
    rows = read_jsonl(path)
    write_jsonl(os.path.join(INPUT, os.path.relpath(path, src)), rows[:LIMIT] if LIMIT else rows)
n_input = sum(len(read_jsonl(p)) for p in jsonl_files(INPUT))
print(f"input: {src} -> {INPUT}, {n_input} instance", flush=True)

# 2. code EMERGE ghim commit + schema quan hệ của EDC (chỉ documents.jsonl, ~12 MB, không cần 400 MB index)
sh(f"git clone -q {EMERGE_REPO} {EMERGE} && git -C {EMERGE} checkout -q {EMERGE_COMMIT}")
sh("pip install -q huggingface_hub")
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

snapshot_download("klimzaporojets/emerge-benchmark", repo_type="dataset", local_dir=f"{EMERGE}/data",
                  allow_patterns=["indices/relik_edc_relation_indexes/*/documents.jsonl"])

# 3. llama-server (giống kaggle/judge/run_judge.py)
cached = find_dir("llama-server")
bin_dir = f"{WORK}/llama-bin"
if cached:
    shutil.copytree(cached, bin_dir, dirs_exist_ok=True)
    sh(f"chmod +x {bin_dir}/llama-server")
else:
    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    libcuda = next((p for p in ("/usr/local/nvidia/lib64/libcuda.so", "/usr/local/cuda/compat/libcuda.so",
                                "/usr/local/cuda/lib64/stubs/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so")
                    if os.path.exists(p)), None)
    assert libcuda, "không tìm thấy libcuda.so"
    sh(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF "
       f"-DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release "
       f"-DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_LIBRARY_PATH={os.path.dirname(libcuda)} "
       f"-DCUDA_cuda_driver_LIBRARY={libcuda}")
    sh(f"cmake --build {WORK}/llama.cpp/build --target llama-server -j $(nproc)")
    os.makedirs(bin_dir, exist_ok=True)
    shutil.copy(f"{WORK}/llama.cpp/build/bin/llama-server", bin_dir)
    shutil.rmtree(f"{WORK}/llama.cpp")

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
    server = subprocess.Popen(cmd, shell=True, stdout=open(f"{WORK}/server.log", "ab"), stderr=subprocess.STDOUT)
    for _ in range(180):
        if healthy() or server.poll() is not None:
            break
        time.sleep(2)
    if healthy():
        break
    print("server không lên, thử bộ cờ khác", flush=True)
    server.kill()
assert healthy(), "llama-server không khởi động được, xem server.log"

# 4. chạy baseline; lỗi ở một baseline không chặn các baseline sau
timing: dict[str, dict] = {}
merge_entries: list[dict] = []
env = {**os.environ, "PYTHONIOENCODING": "utf8", "TOKENIZERS_PARALLELISM": "false"}


def timed(run: str, cmd: str, **kw) -> bool:
    t0 = time.time()
    try:
        sh(f"{cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0", executable="/bin/bash", **kw)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    timing[run] = {"seconds": round(time.time() - t0, 1), "ok": ok}
    return ok


for run in [r for r in RUNS if r in EDC_CONFIGS]:
    cfg = json.load(open(f"{EMERGE}/config/benchmarks/s02_run_benchmarks/{EDC_CONFIGS[run]}/edc_config.json"))
    out = f"{RAW}/{run}/"
    cfg.update(
        llm={"base_url": SERVER, "backend": "http", "model": "local", "concurrency": WORKERS, "timeout": 600,
             "api_key_env": None, "api_version": "v1", "mode": "chat", "model_max_context": None},
        base_input_dir=INPUT + "/", base_input_dir_last_processed=out, base_output_dir=out,
        base_relations_cache_dir=f"{WORK}/edc_cache/", base_target_schema_dir="./data/indices/relik_edc_relation_indexes/",
    )
    cfg_path = f"{WORK}/{run}_edc_config.json"
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    edc_dir = f"{EMERGE}/src/benchmarks/wrappers/edc_plus/edc_tt2kg"
    if timed(run, f"python -u {edc_dir}/s01_run_v3.py --config_file {cfg_path} --batch_size 16 "
                  f"--max_workers {WORKERS}", cwd=EMERGE, env={**env, "PYTHONPATH": edc_dir}):
        merge_entries.append({"dataset_title": run, "input_dataset_path": out, "merge_predictions": True,
                              "predictions_model_name": NAMES[run], "predictions_model_type": "edc",
                              "canonicalize": False})

if "kggen" in RUNS:
    sh("pip install -q kg-gen==0.4.0")
    raw = f"{RAW}/kggen_wrapper"
    # hosted_vllm/: LiteLLM gọi /v1/chat/completions của server OpenAI-compatible; tiền tố openai/ ở bản kg-gen
    # mới hơn bị ép sang Responses API. Nhiệt độ/max_tokens/chunk theo config KGGen Mistral của EMERGE (gpt-5 bị
    # kg-gen ép temp 1.0), max_tokens hạ còn 4096 cho vừa ngữ cảnh 8192/slot.
    wrapper_model = "hosted_vllm/local"
    if timed("kggen", f"python -u {EMERGE}/src/benchmarks/wrappers/kggen/wrapper.py --input {INPUT} --output {raw} "
                      f"--model {wrapper_model} --temperature 0.0 --max-workers {WORKERS} --chunk-size 10240 "
                      f"--base-url {SERVER}/v1 --cache-path {RAW}/kggen_cache.jsonl --max-tokens 4096 "
                      f"--batch-size 1000",
             env={**env, "PYTHONPATH": f"{EMERGE}/src/benchmarks/wrappers", "OPENAI_API_KEY": "none"}):
        # wrapper ghi lại cả instance (kèm prediction cũ) với khóa kg-gen-<model>; merge cần đúng tên chuẩn
        renamed = f"{RAW}/kggen/"
        for path in jsonl_files(raw):
            rows = []
            for r in read_jsonl(path):
                pred = r["predictions"].get(f"kg-gen-{wrapper_model}")
                if pred is None:
                    continue
                pred.update(model=NAMES["kggen"], model_type="kg-gen")
                rows.append({"hash_id": r["hash_id"], "predictions": {NAMES["kggen"]: pred}})
            write_jsonl(os.path.join(renamed, os.path.relpath(path, raw)), rows)
        shutil.rmtree(raw)
        merge_entries.append({"dataset_title": "kggen", "input_dataset_path": renamed, "merge_predictions": True,
                              "predictions_model_name": NAMES["kggen"], "predictions_model_type": "kg-gen"})

karma_runs = [r for r in RUNS if r.startswith("karma_")]
if karma_runs:
    sh(f"git clone -q {KARMA_REPO} {WORK}/karma && git -C {WORK}/karma checkout -q {KARMA_COMMIT}")
    sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {WORK}/kltn")
for run in karma_runs:
    out = f"{RAW}/{run}/"
    if timed(run, f"python -u {WORK}/kltn/scripts/karma_emerge.py --karma {WORK}/karma/karma_pipeline.py "
                  f"--input {INPUT} --output {out} --variant {run.removeprefix('karma_')} --name {NAMES[run]} "
                  f"--base-url {SERVER}/v1 --workers {WORKERS}", env=env):
        merge_entries.append({"dataset_title": run, "input_dataset_path": out, "merge_predictions": True,
                              "predictions_model_name": NAMES[run], "predictions_model_type": "kg-gen"})
server.kill()

# 5. gộp vào evaluation_set bằng bước merge của EMERGE
final = f"{WORK}/evaluation_set/"
merge_cfg = {"input_main_dataset_path": INPUT + "/", "output_merged_dataset_path": final,
             "input_other_datasets_paths": merge_entries}
json.dump(merge_cfg, open(f"{WORK}/merge_config.json", "w"), indent=1)
sh(f"python -u -m merge.s0x_merge_predictions --config_file {WORK}/merge_config.json 2>&1 | tee {LOGS}/merge.log",
   cwd=EMERGE, env={**env, "PYTHONPATH": f"{EMERGE}/src"})

# 6. tóm tắt: số bài có prediction, số triple trung bình, giây/instance (wall / số bài, WORKERS luồng song song)
summary = {"limit": LIMIT, "n_input": n_input, "workers": WORKERS, "runs": {}}
merged = [r for p in jsonl_files(final) for r in read_jsonl(p)]
for run in RUNS:
    name = NAMES[run]
    preds = [r["predictions"][name] for r in merged if name in r.get("predictions", {})]
    sizes = [sum(len(v) for v in p["predicted_triples_oie"].values()) if "predicted_triples_oie" in p
             else len(p.get("predicted_triples", [])) for p in preds]
    t = timing.get(run, {"seconds": None, "ok": False})
    summary["runs"][name] = {
        **t, "n_with_prediction": len(preds), "n_empty": sum(s == 0 for s in sizes),
        "avg_triples": round(sum(sizes) / max(len(sizes), 1), 2),
        "sec_per_instance": round(t["seconds"] / n_input, 2) if t["seconds"] else None,
    }
    stats = [p["karma_stats"] for p in preds if "karma_stats" in p]
    if stats:                         # KARMA: Reader bỏ hết đoạn? REA trích bao nhiêu trước khi Evaluator lọc?
        summary["runs"][name]["karma"] = {
            "reader_dropped_all": sum(s.get("n_relevant_segments", 0) == 0 for s in stats),
            "avg_relationships_before_filter": round(sum(s.get("n_relationships", 0) for s in stats) / len(stats), 2),
            "errors": sum("error" in s for s in stats),
        }
json.dump(summary, open(f"{WORK}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)

for d in (EMERGE, f"{WORK}/karma", f"{WORK}/kltn"):
    shutil.rmtree(d, ignore_errors=True)   # không đưa repo + schema vào output
shutil.rmtree(f"{WORK}/models")       # không đưa 4 GB model vào output
shutil.rmtree(f"{WORK}/input")
print("DONE", flush=True)
