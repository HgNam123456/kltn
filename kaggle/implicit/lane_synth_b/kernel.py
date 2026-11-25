"""Kernel kgu-synth-b (làn GPU): các cấu hình đối chứng trên ChainUpdate (data/chainupdate/chainupdate.jsonl).
Việc 1 (vế sinh như Chương 3): hệ so sánh LLM + lân cận 1-hop (naive, 1.000 mẫu) và pipeline không có bước lọc theo cấu
trúc (llm_nogate, K = 1, 300 mẫu đầu vì số lệnh gọi lớn).
Việc 2 (vế sinh xét cả hai chiều của cạnh vừa thêm, KGU_SYMMETRIC_ADD=1): trần (oracle, K = 1, 2, 3) và pipeline có bước
lọc (llm_gate, K = 2), để đo các sự kiện cả nhóm chuyển (công ty dời trụ sở, phòng ban đổi công ty, CLB đổi giải).
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=synth)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "synth_ctrl", "runner": "implicit", "env": {"IMPLICIT_JOB": "synth", "IMPLICIT_SYNTH_SYM": "0",
                                                         "IMPLICIT_SYNTH_CFGS": "naive,llm_nogate",
                                                         "IMPLICIT_SYNTH_K_NOGATE": "1", "IMPLICIT_SYNTH_LIMIT_NOGATE": "300"}},
    {"name": "synth_sym1", "runner": "implicit", "env": {"IMPLICIT_JOB": "synth", "IMPLICIT_SYNTH_SYM": "1",
                                                         "IMPLICIT_SYNTH_CFGS": "oracle,llm_gate",
                                                         "IMPLICIT_SYNTH_K": "2", "IMPLICIT_SYNTH_K_ORACLE": "1,2,3"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
