"""Chờ kernel Kaggle chạy xong rồi ghi log ra file (không tải output, vì `kaggle kernels output` kéo cả file lớn).

Dùng: PYTHONUTF8=1 python kaggle/kernel_log.py ngocnam2005/kgu-score-emerge results/kaggle/score_v1_kernel.log [--no-wait]
"""
import sys
import time
from pathlib import Path

from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def status(ref: str) -> str:
    return str(api.kernels_status(ref).status)


def fetch_log(ref: str) -> str:
    user, slug = ref.split("/")
    with api.build_kaggle_client() as k:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = user, slug
        out = k.kernels.kernels_api_client.list_kernel_session_output(req)
    return "".join(f"[{e.stream_name}] {e.data}" if hasattr(e, "stream_name") else str(e) for e in out.log)


def main() -> None:
    ref, dest = sys.argv[1], Path(sys.argv[2])
    if "--no-wait" not in sys.argv:
        while not any(s in (st := status(ref)) for s in ("ERROR", "COMPLETE", "CANCEL")):
            print(st, flush=True)
            time.sleep(120)
    print(status(ref), flush=True)
    log = fetch_log(ref)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(log, encoding="utf8")
    tail = log.splitlines()
    print("\n".join(tail[-40:]))
    print(f"-> {dest} ({len(tail)} lines)")


if __name__ == "__main__":
    main()
