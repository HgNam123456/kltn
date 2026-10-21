"""Kernel kgu-atom-edc-webnlg: ATOM trên WebNLG của EDC. Runner chung kaggle/atom_edc/run_atom_edc.py (lấy từ repo)."""
import os
import subprocess

os.environ.update(ATOM_EDC_DATASETS="webnlg", ATOM_EDC_LIMIT="")
KLTN = "/kaggle/working/kltn"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {KLTN}",
               shell=True, check=True)
exec(open(f"{KLTN}/kaggle/atom_edc/run_atom_edc.py", encoding="utf-8").read())
