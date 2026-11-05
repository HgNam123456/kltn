"""Kernel kgu-implicit-tkgqa-trainval (làn GPU): TKGQA train+val, chỉ 744 bài có thay đổi gián tiếp
(data/cascade/tkgqa_trainval_implicit.jsonl), chỉ vế sinh hai chiều (sàn, oracle, Gemma có cổng, Gemma không cổng).
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "tkgqa", "runner": "implicit", "env": {"IMPLICIT_JOB": "tkgqa",
                                                    "IMPLICIT_TKGQA_FILES": "tkgqa_trainval_implicit.jsonl",
                                                    "IMPLICIT_TKGQA_SYMS": "1", "IMPLICIT_TKGQA_SCALE": "2"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
