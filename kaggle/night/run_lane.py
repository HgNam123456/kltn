"""Một làn GPU chạy nối tiếp nhiều việc trong MỘT kernel (chạy qua đêm, không cần ai đẩy kernel tiếp theo).
Stub (kaggle/night/<làn>/kernel.py) đặt LANE_STAGES = JSON danh sách việc, mỗi việc
{"name": thư mục đầu ra, "runner": "extra" | "quality", "env": {EXTRA_DATASET, EXTRA_SYSTEMS, ...}}.
Mỗi việc: lấy code mới nhất của nhánh (sửa lỗi đẩy lên giữa đêm thì việc sau dùng bản sửa), chạy runner trong tiến
trình con với KGU_OUT=/kaggle/working/out/<name> và KGU_DEADLINE = giờ chót chung (runner tự chia giờ cho từng hệ);
xong thì dọn llama-server và thư mục tạm còn sót (runner lỗi giữa chừng) trước việc sau. Giờ chót = lúc bắt đầu +
LANE_HOURS (mặc định 11, dưới giới hạn 12 giờ của Kaggle để output kịp lưu); việc nào bắt đầu khi còn dưới
min_hours (mặc định 1) thì bỏ qua và ghi lại. Trạng thái từng việc: out/lane.json.
"""
import json
import os
import shutil
import signal
import subprocess
import time

STAGES = json.loads(os.environ["LANE_STAGES"])
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
RUNNERS = {"extra": "kaggle/extra_bench/run_extra_bench.py", "quality": "kaggle/quality/run_quality.py"}
WORK = "/kaggle/working"
KLTN = f"{WORK}/kltn"
T0 = time.time()
DEADLINE = T0 + float(os.environ.get("LANE_HOURS", "11")) * 3600
SCRATCH = ("kltn", "edc", "knowledge-net", "data", "models", "llama-bin", "venv_atom", "venv_kggen", "mine")
os.makedirs(f"{WORK}/out", exist_ok=True)
lane = {"stages": [], "deadline_hours": round((DEADLINE - T0) / 3600, 2)}


def save() -> None:
    lane["elapsed_minutes"] = round((time.time() - T0) / 60, 1)
    json.dump(lane, open(f"{WORK}/out/lane.json", "w"), indent=1, ensure_ascii=False)


def cleanup() -> None:
    subprocess.run("pkill -f llama-server; sleep 3; pkill -9 -f llama-server", shell=True)
    for name in os.listdir(WORK):
        if name in SCRATCH or name.startswith("edc_run_") or name.endswith(("_in.txt", "_ref.txt")):
            path = f"{WORK}/{name}"
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)


for st in STAGES:
    rec = {"name": st["name"], "runner": st["runner"], "env": st.get("env", {})}
    lane["stages"].append(rec)
    left_h = (DEADLINE - time.time()) / 3600
    if left_h < st.get("min_hours", 1):
        rec["status"] = f"skipped: còn {left_h:.2f} giờ"
        print(f"### {st['name']}: {rec['status']}", flush=True)
        save()
        continue
    cleanup()
    t0 = time.time()
    try:
        subprocess.run(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}", shell=True, check=True)
        rec["commit"] = subprocess.run(f"git -C {KLTN} rev-parse HEAD", shell=True, capture_output=True,
                                       text=True).stdout.strip()
        env = {**os.environ, **st.get("env", {}), "KGU_OUT": f"{WORK}/out/{st['name']}",
               "KGU_DEADLINE": str(DEADLINE - 600)}
        print(f"### {st['name']} bắt đầu, còn {left_h:.2f} giờ, commit {rec['commit'][:7]}", flush=True)
        save()
        p = subprocess.Popen(["python", "-u", f"{KLTN}/{RUNNERS[st['runner']]}"], env=env, start_new_session=True)
        try:
            rec["returncode"] = p.wait(timeout=max(60, DEADLINE - time.time() - 300))
            rec["status"] = "ok" if rec["returncode"] == 0 else "error"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)      # cả cây tiến trình: EDC, venv, llama-server
            rec["status"] = "killed: chạm giờ chót"
    except Exception as e:
        rec["status"] = f"error: {e!r}"
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    print(f"### {st['name']}: {rec['status']} sau {rec['minutes']} phút", flush=True)
    save()
cleanup()
save()
print(json.dumps(lane, indent=1, ensure_ascii=False), flush=True)
print("DONE", flush=True)
