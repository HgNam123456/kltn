#!/usr/bin/env bash
# Gom dữ liệu EMERGE (evaluation_set + kg_subsets) thành dataset Kaggle `ngocnam2005/kgu-emerge-data`.
# Code thì kernel clone từ GitHub (nhánh feat/core-pipeline), nên chỉ chạy lại khi dữ liệu đổi:
#   bash kaggle/build_kgu_bundle.sh && cd kaggle/dataset_kgu && PYTHONUTF8=1 kaggle datasets version -p . --dir-mode zip -m "<ghi chú>"
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=kaggle/dataset_kgu/bundle
rm -rf "$OUT"
mkdir -p "$OUT"
cp -r data/raw/emerge/evaluation_set data/raw/emerge/kg_subsets "$OUT/"
rm -f "$OUT"/kg_subsets/*.stats.json
cat > kaggle/dataset_kgu/dataset-metadata.json <<'EOF'
{
  "title": "EMERGE data for kgu (evaluation_set + 1-hop KG subsets)",
  "id": "ngocnam2005/kgu-emerge-data",
  "licenses": [{"name": "CC-BY-SA-4.0"}]
}
EOF
du -sh "$OUT"
