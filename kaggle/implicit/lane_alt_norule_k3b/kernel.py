"""Kernel kgu-norule-k3b (làn GPU): ChainUpdate, bỏ bước lọc theo cấu trúc ở vế cắt (llm_gate_add), K = 3, tin 261 tới 499 (tin 500 tới 520 ở kernel kgu-norule-b).
Lần gọi 1 KHÔNG chạy lại: lấy từ tệp seed trong repo, là kết quả lần gọi 1 của lượt llm_gate K = 1 ghép với kết quả LLM
xét các nhóm vế cắt mà bước lọc đã giữ nguyên (kernel kgu-synth-c và kgu-synth-c2). Kernel này chỉ chạy lần gọi 2 và 3;
kết quả của K = 2 lấy từ chính lượt này bằng scripts/score_chainupdate.py --max-round 2.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=synth)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {
        "name": "norule_k3",
        "runner": "implicit",
        "env": {
            "IMPLICIT_JOB": "synth",
            "IMPLICIT_SYNTH_SYM": "0",
            "IMPLICIT_SYNTH_CFGS": "llm_gate_add",
            "IMPLICIT_SYNTH_K": "3",
            "IMPLICIT_SYNTH_SCALE": "3",
            "IMPLICIT_SYNTH_SEED_CALL1": "results/kaggle/implicit/chainupdate_c/seed_call1_norule.jsonl",
            "IMPLICIT_SYNTH_LIMIT": "500",
            "IMPLICIT_SYNTH_OFFSET": "261"
        }
    }
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
