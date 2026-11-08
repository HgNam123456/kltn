"""Kernel kgu-regress (làn GPU): hồi quy nhanh bộ phán sau mỗi lần đổi prompt/cổng — NBA test (438 bài) rồi TKGQA KG cho
trước, mỗi bộ CHỈ cấu hình llm_gate (NBA ~1 giờ, TKGQA ~3 phút LLM) + log phản hồi LLM.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "nba", "runner": "implicit", "env": {"IMPLICIT_JOB": "nba", "IMPLICIT_NBA_CFGS": "llm_gate"}},
    {"name": "given", "runner": "implicit", "env": {"IMPLICIT_JOB": "given", "IMPLICIT_GIVEN_FILES": "tkgqa_test_global.jsonl",
                                                    "IMPLICIT_GIVEN_SYM": "1", "IMPLICIT_GIVEN_CFGS": "llm_gate"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
