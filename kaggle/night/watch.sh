#!/bin/bash
# Chạy trên máy (chỉ gọi Kaggle CLI, không tính toán), tách khỏi phiên Claude qua kaggle/night/watch.ps1.
# 1) Đẩy kgu-night-1 rồi kgu-night-2 ngay khi Kaggle còn chỗ GPU (2 kernel B đang giữ 2 làn): thử đẩy mỗi 2 phút,
#    Kaggle từ chối ("Maximum batch GPU session count") thì đợi lượt sau.
# 2) Khi mọi kernel của đêm xong: tải output về results/kaggle/{covid_repeat/r2,r3, night/lane1,lane2, knet_baseline}
#    rồi dựng bảng tổng hợp results/kaggle/night/REPORT.md (scripts/night_report.py, chỉ đọc JSON).
# Dùng: bash kaggle/night/watch.sh [push|fetch|all]   (mặc định all)
KL=/c/Users/ADMIN/Desktop/kltn
KAGGLE=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/Scripts/kaggle
PY=/c/Users/ADMIN/AppData/Local/Programs/Python/Python311/python
export PYTHONUTF8=1
MODE=${1:-all}
log() { echo "$(date '+%m-%d %H:%M') $*"; }
st() { "$KAGGLE" kernels status "ngocnam2005/$1" 2>&1 | tail -1; }

if [ "$MODE" = all ] || [ "$MODE" = push ]; then
  pending=(lane1 lane2)
  while [ ${#pending[@]} -gt 0 ]; do
    lane=${pending[0]}
    out=$("$KAGGLE" kernels push -p "$KL/kaggle/night/$lane" 2>&1 | tail -1)
    if echo "$out" | grep -qi "success"; then
      log "đẩy $lane: $out"; pending=("${pending[@]:1}"); continue
    fi
    echo "$out" | grep -q "Maximum batch GPU" || log "đẩy $lane lỗi khác: $out"
    sleep 120
  done
fi

if [ "$MODE" = all ] || [ "$MODE" = fetch ]; then
  declare -A dest=([kgu-covid-repeat-r2]=covid_repeat/r2 [kgu-covid-repeat-r3]=covid_repeat/r3
                   [kgu-night-1]=night/lane1 [kgu-night-2]=night/lane2 [kgu-knet-baseline]=knet_baseline)
  for k in kgu-covid-repeat-r2 kgu-covid-repeat-r3 kgu-knet-baseline kgu-night-1 kgu-night-2; do
    sleep 60          # kernel vừa đẩy có thể chưa có trạng thái
    until s=$(st $k) && echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 600; done
    log "$k: $s"
    d="$KL/results/kaggle/${dest[$k]}"; mkdir -p "$d"
    "$KAGGLE" kernels output "ngocnam2005/$k" -p "$d" -o >/dev/null 2>&1 && log "tải $k → ${dest[$k]}" || log "tải $k lỗi"
  done
  cd "$KL" && "$PY" scripts/night_report.py --root results/kaggle --out results/kaggle/night/REPORT.md \
    && log "REPORT.md xong" || log "night_report lỗi"
fi
log "watch xong ($MODE)"
