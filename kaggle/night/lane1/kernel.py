"""Kernel kgu-night-1 (làn GPU 1, chạy qua đêm): NYT11-HRL với cả 4 hệ → KnowledgeNet fold 4 với ATOM+EDC.
Điều phối: kaggle/night/run_lane.py; dữ liệu NYT11 từ dataset riêng tư ngocnam2005/kgu-nyt11."""
import json
import os
import subprocess

os.environ["LANE_STAGES"] = json.dumps([
    {"name": "nyt11", "runner": "extra", "env": {"EXTRA_DATASET": "nyt11", "EXTRA_SYSTEMS": "mine,edc,atom_edc,kggen_edc"}},
    {"name": "knet_atom", "runner": "extra", "env": {"EXTRA_DATASET": "knet", "EXTRA_SYSTEMS": "atom_edc"}},
])
SRC = "/tmp/kltn_lane"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {SRC}",
               shell=True, check=True)
exec(open(f"{SRC}/kaggle/night/run_lane.py", encoding="utf-8").read())
