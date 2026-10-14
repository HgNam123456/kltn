"""Kaggle kernel (GPU T4, không LLM): hiệu chỉnh bộ chấm 2020-COVID-NYT. Chấm 6 cột GPT-4.1 có sẵn trong dataset
(3 lượt trích thẳng, 3 lượt ATOM) bằng nhiều embedder × ngưỡng, chọn cặp tái lập Table 2 của paper ATOM
(F1 0,504 / 0,638). Code + data (data/covid_nyt/2020_covid_nyt.jsonl) lấy từ repo kltn nhánh KGU_BRANCH.
Đầu ra: /kaggle/working/covid_calibration.json.
"""
import subprocess

KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
EMBEDDERS = ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2"]
THRESHOLDS = ["0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9"]
WORK = "/kaggle/working"


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {WORK}/kltn && git -C {WORK}/kltn log --oneline -1")
sh("pip install -q dateparser")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"cd {WORK}/kltn && PYTHONPATH=. python scripts/score_covid_nyt.py --data data/covid_nyt/2020_covid_nyt.jsonl "
   f"--calibrate --embedder {' '.join(EMBEDDERS)} --threshold {' '.join(THRESHOLDS)} --device cuda "
   f"--out {WORK}/covid_calibration.json")
sh(f"rm -rf {WORK}/kltn")
