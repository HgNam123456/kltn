"""Kernel kgu-synth (làn GPU): bộ tự sinh KGU-Synth (data/synth/kgu_synth.jsonl, 1.000 mẫu, thay đổi gián tiếp có nhãn
độ sâu 1–3) với K = 1, 2, 3 lần gọi của thuật toán khoanh vùng đệ quy (--max-calls).
Cấu hình: sàn (none), trần (oracle, K = 1, 2, 3), pipeline có bước lọc theo cấu trúc (llm_gate 0,5, K = 1, 2, 3).
Cuối cùng ghi score_by_depth.{txt,json}. Các cấu hình đối chứng nằm ở kernel kgu-synth-b (lane_synth_b).
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=synth)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "synth", "runner": "implicit", "env": {"IMPLICIT_JOB": "synth", "IMPLICIT_SYNTH_SYM": "0",
                                                    "IMPLICIT_SYNTH_CFGS": "none,oracle,llm_gate",
                                                    "IMPLICIT_SYNTH_K": "1,2,3", "IMPLICIT_SYNTH_K_ORACLE": "1,2,3"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
