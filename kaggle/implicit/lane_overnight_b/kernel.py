"""Kernel kgu-overnight-b (làn GPU, chạy qua đêm): thay đổi gián tiếp theo giao thức KG cho trước trên YAGO-Event
(KG nền đã bổ sung các fact suy ra theo luật: yago_event_aug.jsonl, sinh lại trong kernel). Cấu hình: sàn, trần,
pipeline (gate 0,5), hệ so sánh LLM + lân cận 1-hop (naive).
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=given)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "yago", "runner": "implicit", "env": {"IMPLICIT_JOB": "given", "IMPLICIT_GIVEN_FILES": "yago_event_aug.jsonl",
                                                   "IMPLICIT_GIVEN_SYM": "0", "IMPLICIT_GIVEN_SCALE": "2",
                                                   "IMPLICIT_GIVEN_CFGS": "none,oracle,llm_gate,naive"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
