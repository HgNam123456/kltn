"""Kernel kgu-given-tkgqa-gate (làn GPU): giao thức KG CHO TRƯỚC — TKGQA test, KG nền = toàn bộ fact đang mở của đồ thị toàn cục (13 quan hệ), vế sinh hai chiều. CHỈ cấu hình llm_gate (75 lệnh gọi) + log phản hồi LLM — thăm dò prompt nhanh.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=given)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "given", "runner": "implicit", "env": {"IMPLICIT_JOB": "given", "IMPLICIT_GIVEN_FILES": "tkgqa_test_global.jsonl",
                                                    "IMPLICIT_GIVEN_SYM": "1", "IMPLICIT_GIVEN_SCALE": "1.2", "IMPLICIT_GIVEN_CFGS": "llm_gate"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
