"""Kernel kgu-synth-c (làn GPU): ChainUpdate, K = 1, CHỈ gửi cho LLM các nhóm của vế cắt mà bước lọc theo cấu trúc đã
giữ nguyên không hỏi LLM (mức phụ thuộc dưới ngưỡng 0,5); không xét lại nhóm nào khác. Kết quả được ghép với lượt
llm_gate K = 1 đã chạy (results/kaggle/implicit/chainupdate_a) bằng scripts/score_chainupdate.py --plus.
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
            "IMPLICIT_SYNTH_CFGS": "llm_cut_rejected"
        }
    }
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
