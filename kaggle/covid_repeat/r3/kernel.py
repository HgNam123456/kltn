"""Kernel kgu-covid-repeat-r3: lượt r3 của 2020-COVID-NYT cho bốn hệ (độ ổn định giữa các lượt). Runner chung
kaggle/covid_repeat/run_covid_repeat.py (lấy từ repo)."""
import os
import subprocess

os.environ.update(COVID_REPEAT="r3", COVID_SYSTEMS="mine_v2,atom,atom_direct,kggen", COVID_LIMIT="")
KLTN = "/kaggle/working/kltn"
subprocess.run(f"git clone -q --depth 1 -b feat/core-pipeline https://github.com/HgNam123456/kltn.git {KLTN}",
               shell=True, check=True)
exec(open(f"{KLTN}/kaggle/covid_repeat/run_covid_repeat.py", encoding="utf-8").read())
