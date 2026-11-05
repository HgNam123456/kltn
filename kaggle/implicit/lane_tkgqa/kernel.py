"""Kernel kgu-implicit-tkgqa (làn GPU): TKGQA (M&A, Ong et al. 2023) — thay đổi gián tiếp theo cấu trúc sở hữu.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "tkgqa", "runner": "implicit", "env": {"IMPLICIT_JOB": "tkgqa", "IMPLICIT_TKGQA_FILES": "tkgqa_test.jsonl"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
