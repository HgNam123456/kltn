"""Kernel kgu-synth-c2 (làn GPU): ChainUpdate, bỏ bước lọc theo cấu trúc ở vế cắt. K = 1, CHỈ gửi LLM các nhóm của vế cắt mà bước lọc giữ nguyên, tin 500 tới 999 (nửa sau; nửa đầu ở kgu-synth-c).
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=synth)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {
        "name": "cut_rejected",
        "runner": "implicit",
        "env": {
            "IMPLICIT_JOB": "synth",
            "IMPLICIT_SYNTH_SYM": "0",
            "IMPLICIT_SYNTH_CFGS": "llm_cut_rejected",
            "IMPLICIT_SYNTH_OFFSET": "500"
        }
    }
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
