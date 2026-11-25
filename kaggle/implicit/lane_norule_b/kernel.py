"""Kernel kgu-norule-b (làn GPU): ChainUpdate, bỏ bước lọc theo cấu trúc ở vế cắt. Một kernel làm trọn chuỗi trên các tin 261 tới 520: K = 1 chỉ các nhóm vế cắt bị bước lọc giữ nguyên, ghép với lượt llm_gate K = 1 đã có để được kết quả lần gọi 1, rồi chạy tiếp lần gọi 2 và 3 từ đó.
Điều phối: kaggle/night/run_lane.py (runner "implicit" → kaggle/implicit/run_implicit.py, IMPLICIT_JOB=norule)."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {
        "name": "norule",
        "runner": "implicit",
        "env": {
            "IMPLICIT_JOB": "norule",
            "NORULE_LIMIT": "521",
            "NORULE_OFFSET": "261"
        }
    }
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
