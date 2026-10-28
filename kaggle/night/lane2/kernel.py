"""Kernel kgu-night-2 (làn GPU 2, chạy qua đêm): độ chính xác (judge) + độ gọn trên output COVID-NYT / MINE-1 đã có
→ KnowledgeNet fold 4 với pipeline của mình, EDC, KGGen+EDC. Điều phối: kaggle/night/run_lane.py."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "quality", "runner": "quality"},
    {"name": "knet_main", "runner": "extra", "env": {"EXTRA_DATASET": "knet", "EXTRA_SYSTEMS": "mine,edc,kggen_edc"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
