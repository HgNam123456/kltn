"""Kaggle kernel: chấm prediction của mình bằng bộ chấm chính chủ EMERGE (Completeness, G-BERTScore, QID exact
match) trên GPU, không nạp KG snapshot, trên TOÀN BỘ 3.500 bài (bản dev350 là kaggle/emerge_eval/run_eval.py; khác ở
dataset và MODELS: chỉ các hệ thống trong bảng báo cáo, để vừa thời gian GPU).

Đầu vào: dataset ngocnam2005/emerge-full-mine = evaluation_set đủ 3.500 bài ra từ kernel kgu-baselines-full (prediction
16 baseline của EMERGE + EDC+ ICL, KGGen chạy lại bằng Gemma 4 E4B) cộng prediction của mình
(scripts/export_emerge_predictions.py, Exists/Deprecate từ emerge_full_v3_kaggle, Add v2 từ emerge_full_add_v2_kaggle).
Đầu ra: /kaggle/working/eval.log (bảng), /kaggle/working/out/wiki_eval_result.pkl.
"""
import json
import os
import subprocess

def find_dataset(root: str = "/kaggle/input") -> str:
    """Kaggle có thể giải nén zip vào gốc hoặc vào thư mục con; tìm thư mục chứa snapshot_*/."""
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.startswith("snapshot_") for d in dirnames):
            return dirpath + "/"
    raise SystemExit(f"no snapshot_* dir under {root}")


DATASET = find_dataset()
MINE = "mine/gemma4-e4b"
MODELS = [
    MINE,
    "edc-plus-llamacpp/gemma-4-e4b", "kg-gen/llamacpp/gemma-4-e4b",           # chạy lại bằng Gemma (kgu-baselines-full)
    "kg-aware/gpt-5.1/oracle", "edc-plus-open-ai/gpt-5.1/non-canonicalized",  # bản GPT-5.1 của EMERGE để đối chiếu
    "edc-plus-zshot-open_ai/GPT-5_1", "kg-gen/azure/gpt5.1",
]
QID_MODELS = [MINE, "kg-aware/gpt-5.1/oracle"]


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("git clone --depth 1 https://github.com/klimzaporojets/emerge.git /kaggle/working/emerge")
os.chdir("/kaggle/working/emerge")
sh("pip install -q bert_score rouge_score sentence-transformers typed-argument-parser python-Levenshtein "
   "krippendorff lz4 'git+https://github.com/klimzaporojets/unified-llm-client.git@3698d79'")
sh("python -c \"import nltk; [nltk.download(x, quiet=True) for x in ('words', 'punkt', 'punkt_tab')]\"")

base = "config/evaluation/s0x_evaluate_predictions/20260324_all_models_with_zs_fixed_with_kg/config.json"
cfg = json.load(open(base))
cfg.update(
    input_dataset_path=DATASET,
    output_path="/kaggle/working/out/wiki_eval_result.pkl",
    cache_path="/kaggle/working/out/wiki_eval_result.pkl",
    load_results_from_cache=False,
    snapshot_year_to_kg_file={y: None for y in cfg["snapshot_year_to_kg_file"]},   # không nạp KG 22 GB
    models_to_evaluate=MODELS,
    models_to_report=MODELS,
)
mc = cfg["metrics_to_calculate"]
mc.pop("entity_coverage", None)                                    # nặng, không có trong bảng chính
for m in mc["completeness"]["similarity_models"] + mc["graph_judge"]["similarity_models"]:
    m["workers"] = min(m.get("workers", 4), 4)
mc["graph_judge"]["similarity_bleu_rouge"]["workers"] = 4
mc["cie_exact_match"]["models_to_report"] = QID_MODELS
os.makedirs("/kaggle/working/out", exist_ok=True)
os.makedirs("logs", exist_ok=True)
json.dump(cfg, open("/kaggle/working/out/config.json", "w"), indent=1)

sh("PYTHONPATH=$PWD/src python -u -m evaluation.s0x_evaluate_predictions "
   "--config_file /kaggle/working/out/config.json 2>&1 | tee /kaggle/working/eval.log")
sh("rm -rf /kaggle/working/emerge")     # không đưa repo vào output
print("DONE")
