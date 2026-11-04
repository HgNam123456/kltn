#!/bin/bash
# Tải output + log của hai kernel kgu-implicit-* về results/kaggle/implicit/ (chạy trên máy, chỉ gọi Kaggle CLI).
# Dùng: bash kaggle/implicit/fetch.sh   (chờ tới khi cả hai xong; Task Scheduler "kgu-implicit-fetch" gọi lúc 06:30)
KL=/c/Users/ADMIN/Desktop/kltn
KAGGLE=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/Scripts/kaggle
PY=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/python
export PYTHONUTF8=1
log() { echo "$(date '+%m-%d %H:%M') $*"; }
st() { "$KAGGLE" kernels status "ngocnam2005/$1" 2>&1 | tail -1; }
for k in kgu-implicit-cascade kgu-implicit-elken; do
  until s=$(st $k) && echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 600; done
  log "$k: $s"
  d="$KL/results/kaggle/implicit/${k#kgu-implicit-}"; mkdir -p "$d"
  "$KAGGLE" kernels output "ngocnam2005/$k" -p "$d" -o >/dev/null 2>&1 && log "tải $k → $d" || log "tải $k lỗi"
  cd "$KL" && "$PY" kaggle/kernel_log.py "ngocnam2005/$k" "$d/kernel.log" --no-wait >/dev/null 2>&1 && log "log $k xong"
done
log "fetch xong"
