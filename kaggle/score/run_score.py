"""Kaggle kernel: chấm mềm (C, G-BERTScore-R viết lại từ EMERGE) cho file kết quả của run_emerge.py, kèm baseline
kg-prompt trong dataset để đối chiếu với bảng chính thức.

Đầu vào: code từ GitHub HgNam123456/kltn (nhánh BRANCH); dataset ngocnam2005/kgu-emerge-data (evaluation_set +
         kg_subsets); dataset ngocnam2005/kgu-results (các jsonl của run_emerge.py, do kaggle/dataset_results/ gửi).
Đầu ra: /kaggle/working/results/*.soft.json + score.log. Sửa JOBS để chọn cặp (judge, add) cần chấm.
"""
import os
import subprocess

# (jsonl Exists/Deprecate, jsonl Add hoặc None, tên file kết quả)
JOBS = [
    ("emerge_dev350_v3_softyear.jsonl", "emerge_dev350_add_v1.jsonl", "emerge_dev350_v3_addv1.soft.json"),
    ("emerge_dev350_v3_softyear.jsonl", "emerge_dev350_add_v2.jsonl", "emerge_dev350_v3_addv2.soft.json"),
]
BASELINES = "kg-aware/gpt-5.1/oracle"
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


bundle = f"{WORK}/kltn"
sh(f"git clone --depth 1 -b {BRANCH} {REPO} {bundle}")
data = find_dir("kg_subsets")
assert data, "không thấy dataset kgu-emerge-data trong /kaggle/input"
os.makedirs(f"{bundle}/data/raw", exist_ok=True)
os.symlink(data, f"{bundle}/data/raw/emerge")
results = find_dir(JOBS[0][0])
assert results, "không thấy dataset kgu-results trong /kaggle/input"
print("data:", data, "results:", results, flush=True)

sh("pip install -q sentence-transformers bert_score scipy pydantic")
os.makedirs(f"{WORK}/results", exist_ok=True)
env = {**os.environ, "PYTHONPATH": bundle, "PYTHONIOENCODING": "utf8"}
for judge, add, out in JOBS:
    args = f"--judge {results}/{judge}" + (f" --add {results}/{add}" if add else "")
    sh(f"cd {bundle} && python scripts/score_emerge_soft.py {args} --baselines {BASELINES} --device cuda "
       f"--out {WORK}/results/{out} 2>&1 | tee -a {WORK}/score.log", env=env)
sh(f"rm -rf {bundle}")
print("DONE", flush=True)
