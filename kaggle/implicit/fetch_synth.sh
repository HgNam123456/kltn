#!/bin/bash
# Tải output + log của hai kernel KGU-Synth (kgu-synth, kgu-synth-b) về results/kaggle/implicit/synth_a|b.
# Dùng: bash kaggle/implicit/fetch_synth.sh   (chờ tới khi từng kernel xong; chỉ gọi Kaggle CLI, không chạy gì trên máy)
KL=/c/Users/ADMIN/Desktop/kltn
KAGGLE=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/Scripts/kaggle
PY=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/python
export PYTHONUTF8=1
log() { echo "$(date '+%m-%d %H:%M') $*"; }
st() { "$KAGGLE" kernels status "ngocnam2005/$1" 2>&1 | tail -1; }
for pair in "kgu-synth-b synth_b" "kgu-synth synth_a"; do
  set -- $pair
  until s=$(st $1) && echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 300; done
  log "$1: $s"
  d="$KL/results/kaggle/implicit/$2"; mkdir -p "$d"
  "$KAGGLE" kernels output "ngocnam2005/$1" -p "$d" -o >/dev/null 2>&1 && log "tải $1 → $d" || log "tải $1 lỗi"
  cd "$KL" && "$PY" kaggle/kernel_log.py "ngocnam2005/$1" "$d/kernel.log" --no-wait >/dev/null 2>&1 && log "log $1 xong"
done
log "fetch xong"
