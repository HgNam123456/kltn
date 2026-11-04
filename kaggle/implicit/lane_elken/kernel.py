"""Kernel kgu-implicit-elken (làn GPU, chạy qua đêm): benchmark thay đổi gián tiếp — IMPLICIT_JOB=elken.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "elken", "runner": "implicit", "env": {"IMPLICIT_JOB": "elken", "IMPLICIT_PROMPT": "v2"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
