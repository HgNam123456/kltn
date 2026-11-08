"""Runner kernel "implicit": các benchmark đo THAY ĐỔI GIÁN TIẾP của KG sau cập nhật, Gemma 4 E4B qua llama-server
CUDA (như kaggle/judge/run_judge.py). Chạy qua kaggle/night/run_lane.py (runner "implicit") hoặc trực tiếp.

IMPLICIT_JOB = "cascade": ICEWS14-Event + YAGO-Event sinh lại từ luật CascadeKG (scripts/prep_cascade.py, dữ liệu thô
    tải từ GitHub ngay trong kernel), rồi scripts/run_eval.py --jsonl với các cấu hình: gold_explicit + none (sàn),
    + oracle (trần của khoanh vùng), + llm gate 0,5 (như NBA), + llm không cổng. Mỗi lượt LLM có giờ chót riêng.
IMPLICIT_JOB = "nba": hồi quy NBAtransactions test (dataset ngocnam2005/kgu-nba): IMPLICIT_NBA_CFGS (mặc định none,oracle,llm_gate),
    IMPLICIT_NBA_SCALE nhân giờ LLM. Mốc 09/2026: gate 0.5 F1 0,993 / add 0,840 / del 0,997, 1.428 lệnh gọi.
IMPLICIT_JOB = "elken": ELKEN (clone repo THU-KEG) → scripts/run_elken.py trên test rồi train.
IMPLICIT_JOB = "given": giao thức "KG cho trước" (toàn bộ thực thể/triple đã biết là KG nền, tập quan hệ ĐÓNG = quan hệ
    trong KG nền, không gộp nhãn): IMPLICIT_GIVEN_FILES = jsonl định dạng header base + chênh lệch trong data/cascade
    (tkgqa_test_global.jsonl, icews14_event_global.jsonl có trong repo; yago_event_global.jsonl 63 MB được sinh lại
    bằng prep_cascade.py --global nếu thiếu); IMPLICIT_GIVEN_SYM = KGU_SYMMETRIC_ADD; IMPLICIT_GIVEN_SCALE nhân giờ LLM;
    IMPLICIT_GIVEN_CFGS = tập con của none,oracle,llm_gate,llm_nogate. Mỗi lượt LLM ghi logs/<name>.llm.jsonl (KGU_LLM_LOG).
Đầu ra: $KGU_OUT/<job>/... + summary.json; server.log. KGU_DEADLINE (epoch) = giờ chót chung.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

JOB = os.environ.get("IMPLICIT_JOB", "cascade")
WORKERS = os.environ.get("IMPLICIT_WORKERS", "4")
N_ICEWS, N_YAGO = os.environ.get("IMPLICIT_N_ICEWS", "300"), os.environ.get("IMPLICIT_N_YAGO", "400")
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
ELKEN_REPO = "https://github.com/THU-KEG/Event-Level-Knowledge-Editing.git"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
WORK = "/kaggle/working"
ROOT = os.environ.get("KGU_OUT", f"{WORK}/out")
OUT = f"{ROOT}/{JOB}"
KLTN = f"{WORK}/kltn"
T0 = time.time()
DEADLINE = float(os.environ.get("KGU_DEADLINE") or T0 + 11 * 3600)
summary: dict = {"job": JOB, "runs": {}, "notes": []}


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


def save() -> None:
    summary["elapsed_minutes"] = round((time.time() - T0) / 60, 1)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)


def timed(name: str, cmd: str, env: dict, max_h: float = 4.0, reserve_min: float = 15) -> bool:
    """Chạy một lượt với giờ chót = min(max_h, giờ chót chung − reserve). Hết giờ: giết, ghi lại, chạy tiếp lượt sau."""
    left = DEADLINE - time.time() - reserve_min * 60
    if left < 120:
        summary["runs"][name] = {"status": f"skipped: còn {left / 60:.0f} phút"}
        save()
        return False
    budget = min(max_h * 3600, left)
    print(f"### {name}: tối đa {budget / 60:.0f} phút\n$ {cmd}", flush=True)
    t0 = time.time()
    log = open(f"{OUT}/logs/{name}.log", "w")
    p = subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = p.wait(timeout=budget)
        status = "ok" if rc == 0 else f"error rc={rc}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        status = "killed: hết giờ"
    log.close()
    summary["runs"][name] = {"status": status, "minutes": round((time.time() - t0) / 60, 1)}
    print(f"### {name}: {status} sau {summary['runs'][name]['minutes']} phút", flush=True)
    save()
    return status == "ok"


os.makedirs(f"{OUT}/logs", exist_ok=True)
summary["gpus"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
if not os.path.exists(KLTN):
    sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q huggingface_hub pydantic 'openai>=1' tqdm")

# 1. llama-server (binary cache) + model. Ảnh Kaggle 11/2026 (Python 3.13) không còn libcudart.so.12 mà binary
# cache (build 09/2026) cần → cài CUDA 12 runtime từ pip và trỏ LD_LIBRARY_PATH; nếu ldd vẫn báo thiếu thì build lại.
cached = find_dir("llama-server")
assert cached, "thiếu dataset ngocnam2005/llama-server-cuda"
bin_dir = f"{WORK}/llama-bin"
shutil.copytree(cached, bin_dir, dirs_exist_ok=True)
sh(f"chmod +x {bin_dir}/llama-server")
sh("pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12")
import glob  # noqa: E402

LD_DIRS = {os.path.dirname(f) for pat in ("/usr/local/lib/python3*/dist-packages/nvidia/**/*.so*",
                                           "/usr/lib/python3*/dist-packages/nvidia/**/*.so*",
                                           "/usr/local/lib/python3*/site-packages/nvidia/**/*.so*")
           for f in glob.glob(pat, recursive=True)} | {"/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"}
LD = ":".join(sorted(LD_DIRS))
os.environ["LD_LIBRARY_PATH"] = LD + ":" + os.environ.get("LD_LIBRARY_PATH", "")
missing = subprocess.run(f"ldd {bin_dir}/llama-server | grep 'not found'", shell=True, capture_output=True, text=True).stdout
summary["ldd_missing_after_pip"] = missing.strip()
print("LD_LIBRARY_PATH:", LD, "| ldd thiếu:", missing or "(không)", flush=True)
if missing:
    summary["notes"].append("binary cache thiếu thư viện → build lại llama.cpp trong kernel")
    sh(f"rm -rf {WORK}/llama.cpp && git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    libcuda = next((p for p in ("/usr/local/nvidia/lib64/libcuda.so", "/usr/local/cuda/compat/libcuda.so",
                                "/usr/local/cuda/lib64/stubs/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so",
                                "/usr/lib/x86_64-linux-gnu/libcuda.so.1") if os.path.exists(p)), None)
    assert libcuda, "không tìm thấy libcuda.so"
    sh(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF "
       f"-DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release "
       f"-DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_LIBRARY_PATH={os.path.dirname(libcuda)} "
       f"-DCUDA_cuda_driver_LIBRARY={libcuda} 2>&1 | tail -5")
    sh(f"cmake --build {WORK}/llama.cpp/build --target llama-server -j $(nproc) 2>&1 | tail -3")
    shutil.copy(f"{WORK}/llama.cpp/build/bin/llama-server", f"{bin_dir}/llama-server")
    shutil.rmtree(f"{WORK}/llama.cpp")
    os.makedirs(f"{OUT}/llama-bin", exist_ok=True)
    shutil.copy(f"{bin_dir}/llama-server", f"{OUT}/llama-bin/llama-server")      # để cập nhật dataset cache
from huggingface_hub import hf_hub_download  # noqa: E402

model = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=f"{WORK}/models")
draft = hf_hub_download(GGUF_REPO, MTP_FILE, local_dir=f"{WORK}/models")
COMMON = (f"CUDA_VISIBLE_DEVICES=0 {bin_dir}/llama-server -m {model} --alias local --host 127.0.0.1 --port 8080 -ngl 99 "
          f"--jinja --temp 0")
ATTEMPTS = [
    COMMON + f" --parallel {WORKERS} --cont-batching --kv-unified-per-slot 8192 --reasoning off --flash-attn on"
             f" --spec-type draft-mtp --spec-draft-n-max 2 --model-draft {draft}",
    COMMON + f" --parallel {WORKERS} --cont-batching -c {8192 * int(WORKERS)} --flash-attn on",
]


def healthy() -> bool:
    try:
        return b"ok" in urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).read()
    except Exception:
        return False


server = None
for cmd in ATTEMPTS:
    print(f"$ {cmd}", flush=True)
    server = subprocess.Popen(cmd, shell=True, stdout=open(f"{OUT}/server.log", "ab"), stderr=subprocess.STDOUT)
    for _ in range(180):
        if healthy() or server.poll() is not None:
            break
        time.sleep(2)
    if healthy():
        break
    server.kill()
if not healthy():
    print(open(f"{OUT}/server.log", "rb").read()[-3000:].decode("utf8", "replace"), flush=True)
    raise SystemExit("llama-server không khởi động được, xem server.log")
summary["server_cmd"] = cmd
save()

ENV = {**os.environ, "PYTHONPATH": KLTN, "PYTHONIOENCODING": "utf8", "PYTHONUTF8": "1",
       "KGU_LLM_BASE_URL": "http://127.0.0.1:8080/v1", "KGU_LLM_MODEL": "local", "KGU_LLM_API_KEY": "none",
       "KGU_LLM_EXTRA_BODY": "{}"}

# 2. việc
if JOB == "cascade":
    data = f"{OUT}/data"
    sh(f"cd {KLTN} && python scripts/prep_cascade.py --raw {WORK}/data/raw/cascade --out {data} "
       f"--n-icews {N_ICEWS} --n-yago {N_YAGO} 2>&1 | tee {OUT}/logs/prep.log")
    summary["prep_stats"] = json.load(open(f"{data}/prep_stats.json"))
    save()
    for ds in ("icews14_event", "yago_event"):
        for cfg, extra, max_h in (("none", "", 0.5), ("oracle", "", 0.5),
                                  ("llm_gate", "--judge llm --judge-batch 1 --judge-gate 0.5", 2.5),
                                  ("llm_nogate", "--judge llm --judge-batch 1", 2.5)):
            judge = "--judge " + cfg if cfg in ("none", "oracle") else extra
            name = f"{ds}_goldexp_{cfg}"
            timed(name, f"cd {KLTN} && python -u scripts/run_eval.py --jsonl {data}/{ds}.jsonl --extractor gold_explicit "
                        f"{judge} --workers {WORKERS} --out {OUT}/{name}.jsonl", ENV, max_h=max_h)
elif JOB == "tkgqa":
    # dữ liệu đã dựng sẵn trong repo (scripts/prep_tkgqa.py); IMPLICIT_TKGQA_FILES = danh sách jsonl trong data/cascade
    files = os.environ.get("IMPLICIT_TKGQA_FILES", "tkgqa_test.jsonl").split(",")
    syms = os.environ.get("IMPLICIT_TKGQA_SYMS", "1,0").split(",")       # "1" = vế sinh hai chiều, "0" = một chiều
    scale = float(os.environ.get("IMPLICIT_TKGQA_SCALE", "1"))          # nhân ngân sách giờ của lượt LLM (tập lớn)
    for fn in files:
        ds = fn.replace(".jsonl", "")
        for sym in syms:
            for cfg, judge, max_h in (("none", "--judge none", 0.5), ("oracle", "--judge oracle", 0.5),
                                      ("llm_gate", "--judge llm --judge-batch 1 --judge-gate 0.5", 2.0 * scale),
                                      ("llm_nogate", "--judge llm --judge-batch 1", 2.5 * scale)):
                if sym == "0" and cfg == "llm_gate":
                    continue                                   # vế sinh một chiều phủ 10%: chỉ chạy nogate để đối chứng
                name = f"{ds}_sym{sym}_goldexp_{cfg}"
                timed(name, f"cd {KLTN} && python -u scripts/run_eval.py --jsonl {KLTN}/data/cascade/{fn} "
                            f"--extractor gold_explicit {judge} --workers {WORKERS} --out {OUT}/{name}.jsonl",
                      {**ENV, "KGU_SYMMETRIC_ADD": sym}, max_h=max_h)
elif JOB == "given":
    files = os.environ.get("IMPLICIT_GIVEN_FILES", "tkgqa_test_global.jsonl").split(",")
    sym = os.environ.get("IMPLICIT_GIVEN_SYM", "0")
    scale = float(os.environ.get("IMPLICIT_GIVEN_SCALE", "1"))
    data = f"{KLTN}/data/cascade"
    if any(not os.path.exists(f"{data}/{fn}") for fn in files):
        sh(f"cd {KLTN} && python scripts/prep_cascade.py --global --raw {WORK}/data/raw/cascade --out {data} "
           f"2>&1 | tee {OUT}/logs/prep.log")
        summary["prep_stats"] = json.load(open(f"{data}/prep_stats_global.json"))
        save()
    cfgs = os.environ.get("IMPLICIT_GIVEN_CFGS", "none,oracle,llm_gate,llm_nogate").split(",")
    for fn in files:
        ds = fn.replace(".jsonl", "")
        for cfg, judge, max_h in (("none", "--judge none", 1.0), ("oracle", "--judge oracle", 1.0),
                                  ("llm_gate", "--judge llm --judge-batch 1 --judge-gate 0.5", 2.5 * scale),
                                  ("llm_nogate", "--judge llm --judge-batch 1", 2.5 * scale)):
            if cfg not in cfgs:
                continue
            name = f"{ds}_sym{sym}_goldexp_{cfg}"
            timed(name, f"cd {KLTN} && python -u scripts/run_eval.py --jsonl {data}/{fn} --extractor gold_explicit "
                        f"{judge} --workers {WORKERS} --out {OUT}/{name}.jsonl",
                  {**ENV, "KGU_SYMMETRIC_ADD": sym, "KGU_LLM_LOG": f"{OUT}/logs/{name}.llm.jsonl"}, max_h=max_h)
elif JOB == "nba":
    # hồi quy trên benchmark neo NBAtransactions (test 438 bài) sau khi đổi prompt bộ phán; dữ liệu từ dataset kgu-nba
    nba = find_dir("NBAtransactions_test.json")
    assert nba, "không thấy dataset kgu-nba trong /kaggle/input"
    cfgs = os.environ.get("IMPLICIT_NBA_CFGS", "none,oracle,llm_gate").split(",")
    scale = float(os.environ.get("IMPLICIT_NBA_SCALE", "1"))
    for cfg, judge, max_h in (("none", "--judge none", 0.5), ("oracle", "--judge oracle", 0.5),
                              ("llm_gate", "--judge llm --judge-batch 1 --judge-gate 0.5", 3.0 * scale),
                              ("llm_nogate", "--judge llm --judge-batch 1", 3.0 * scale)):
        if cfg not in cfgs:
            continue
        name = f"nba_test_goldexp_{cfg}"
        timed(name, f"cd {KLTN} && python -u scripts/run_eval.py --data-dir {nba} --split test --extractor gold_explicit "
                    f"{judge} --workers {WORKERS} --out {OUT}/{name}.jsonl",
              {**ENV, "KGU_LLM_LOG": f"{OUT}/logs/{name}.llm.jsonl"}, max_h=max_h)
elif JOB == "elken":
    elken = f"{WORK}/elken"
    sh(f"git clone -q --depth 1 {ELKEN_REPO} {elken}")
    prompt = os.environ.get("IMPLICIT_PROMPT", "v1")
    for split, max_h in (("test", 3.0), ("train", 3.0)):
        name = f"elken_{split}" + ("" if prompt == "v1" else f"_{prompt}")
        timed(name, f"cd {KLTN} && python -u scripts/run_elken.py --elken {elken} --split {split} --workers {WORKERS} "
                    f"--prompt {prompt} --out {OUT}/{name}.jsonl", ENV, max_h=max_h)
else:
    raise SystemExit(f"IMPLICIT_JOB lạ: {JOB}")

# 3. gom summary của từng lượt
for f in sorted(os.listdir(OUT)):
    if f.endswith(".summary.json"):
        s = json.load(open(f"{OUT}/{f}"))
        summary["runs"].setdefault(f.replace(".summary.json", ""), {})["summary"] = {
            k: v for k, v in s.items() if k != "by_event_type"}
save()
server.kill()
shutil.rmtree(f"{WORK}/models", ignore_errors=True)      # không đưa 4 GB model vào output
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
print("DONE", flush=True)
