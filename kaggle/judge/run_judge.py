"""Kaggle kernel: phục vụ Gemma 4 E4B (GGUF QAT) bằng llama-server CUDA rồi chạy scripts/run_emerge.py của kgu.

Đầu vào: code clone từ GitHub HgNam123456/kltn (nhánh BRANCH); dataset ngocnam2005/kgu-emerge-data
         (evaluation_set + kg_subsets, do kaggle/build_kgu_bundle.sh sinh); tùy chọn dataset
         ngocnam2005/llama-server-cuda (binary đã build, bỏ qua bước build ~15 phút).
Đầu ra: /kaggle/working/results/*.jsonl + .summary.json, /kaggle/working/llama-bin/ (binary để tạo dataset cache),
        /kaggle/working/server.log.
Sửa RUNS để chọn lượt chạy.
"""
import glob
import os
import shutil
import subprocess
import time
import urllib.request

RUNS = [
    ["--batch-size", "12", "--out", "results/emerge_dev350_v3_kaggle.jsonl"],
    ["--judge", "add", "--out", "results/emerge_dev350_add_v1_kaggle.jsonl"],
]
WORKERS = "4"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
WORK = "/kaggle/working"
REPO, BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


# 1. code từ GitHub, data từ dataset (symlink vào data/raw/emerge như trên máy)
bundle = f"{WORK}/kltn"
sh(f"git clone --depth 1 -b {BRANCH} {REPO} {bundle}")
data = find_dir("kg_subsets")
assert data, "không thấy dataset kgu-emerge-data trong /kaggle/input"
os.makedirs(f"{bundle}/data/raw", exist_ok=True)
os.symlink(data, f"{bundle}/data/raw/emerge")
print("data:", data, flush=True)

# 2. llama-server: dùng binary cache nếu có, không thì build CUDA
cached = find_dir("llama-server")
bin_dir = f"{WORK}/llama-bin"
if cached:
    shutil.copytree(cached, bin_dir, dirs_exist_ok=True)
    sh(f"chmod +x {bin_dir}/llama-server")
else:
    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    # Ảnh Kaggle không có libcuda.so ở chỗ FindCUDAToolkit tìm → CMake báo thiếu CUDA::cuda_driver.
    # Trỏ vào stubs của toolkit (và liệt kê để chẩn đoán nếu vẫn lỗi).
    sh("find / -name 'libcuda.so*' -not -path '*/proc/*' 2>/dev/null | head; ls /usr/local/cuda/lib64/stubs | head")
    stubs = "/usr/local/cuda/lib64/stubs"
    sh(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF "
       f"-DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release "
       f"-DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_LIBRARY_PATH={stubs} -DCUDA_cuda_driver_LIBRARY={stubs}/libcuda.so")
    sh(f"cmake --build {WORK}/llama.cpp/build --target llama-server -j $(nproc)")
    os.makedirs(bin_dir, exist_ok=True)
    shutil.copy(f"{WORK}/llama.cpp/build/bin/llama-server", bin_dir)
    shutil.rmtree(f"{WORK}/llama.cpp")

# 3. model
sh("pip install -q huggingface_hub pydantic openai tqdm")
from huggingface_hub import hf_hub_download  # noqa: E402

model = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=f"{WORK}/models")
draft = hf_hub_download(GGUF_REPO, MTP_FILE, local_dir=f"{WORK}/models")

# 4. server (giống scripts/llm_server.ps1; nếu cờ mới không hợp thì rơi về bộ cờ tối thiểu)
COMMON = f"{bin_dir}/llama-server -m {model} --alias local --host 127.0.0.1 --port 8080 -ngl 99 --jinja --temp 0"
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
    log = open(f"{WORK}/server.log", "ab")
    server = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if healthy() or server.poll() is not None:
            break
        time.sleep(2)
    if healthy():
        break
    print("server không lên, thử bộ cờ khác", flush=True)
    server.kill()
assert healthy(), "llama-server không khởi động được, xem server.log"

# 5. chạy các lượt
env = {**os.environ, "PYTHONPATH": bundle, "PYTHONIOENCODING": "utf8",
       "KGU_LLM_BASE_URL": "http://127.0.0.1:8080/v1", "KGU_LLM_MODEL": "local", "KGU_LLM_API_KEY": "none"}
os.makedirs(f"{WORK}/results", exist_ok=True)
for args in RUNS:
    out = args[args.index("--out") + 1]
    args = [a if a != out else f"{WORK}/{out}" for a in args]
    sh(f"cd {bundle} && python scripts/run_emerge.py --workers {WORKERS} " + " ".join(args)
       + f" 2>&1 | tee {WORK}/results/{os.path.basename(out)}.log", env=env)
shutil.rmtree(bundle)                 # không đưa repo vào output

server.kill()
for f in glob.glob(f"{WORK}/models/*"):
    os.remove(f)                      # không đưa 4 GB model vào output
print("DONE", flush=True)
