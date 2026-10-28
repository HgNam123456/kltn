"""Kernel kgu-extra-knet-a: KnowledgeNet fold 4 với các hệ mine,edc. Runner chung kaggle/extra_bench/run_extra_bench.py
(lấy từ repo); dữ liệu tải từ GitHub diffbot/knowledge-net trong kernel (scripts/prep_knet.py)."""
import os
import subprocess

os.environ.update(EXTRA_DATASET="knet", EXTRA_SYSTEMS="mine,edc", EXTRA_LIMIT="")
KLTN = "/kaggle/working/kltn"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {KLTN}",
               shell=True, check=True)
exec(open(f"{KLTN}/kaggle/extra_bench/run_extra_bench.py", encoding="utf-8").read())
