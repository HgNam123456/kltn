"""Kernel kgu-nba-regress (làn GPU): hồi quy bộ phán LLM trên NBAtransactions test (438 bài) sau khi đổi prompt vế sinh
(thực đơn lựa chọn hợp lệ, chiều thừa hưởng theo số cạnh, quy luật toàn KG). Mốc 09/2026: gate 0.5 F1 0,993 / add 0,840 / del 0,997.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=nba)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "nba", "runner": "implicit", "env": {"IMPLICIT_JOB": "nba", "IMPLICIT_NBA_CFGS": "none,oracle,llm_gate",
                                                  "IMPLICIT_NBA_SCALE": "1"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
