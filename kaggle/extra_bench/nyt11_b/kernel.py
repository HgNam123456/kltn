"""Kernel kgu-extra-nyt11-b: NYT11-HRL với các hệ atom_edc,kggen_edc. Runner chung kaggle/extra_bench/run_extra_bench.py (lấy từ
repo); dữ liệu từ dataset riêng tư ngocnam2005/kgu-nyt11 (scripts/prep_nyt11.py)."""
import os
import subprocess

os.environ.update(EXTRA_DATASET="nyt11", EXTRA_SYSTEMS="atom_edc,kggen_edc", EXTRA_LIMIT="")
KLTN = "/kaggle/working/kltn"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {KLTN}",
               shell=True, check=True)
exec(open(f"{KLTN}/kaggle/extra_bench/run_extra_bench.py", encoding="utf-8").read())
