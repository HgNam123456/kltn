"""Kernel kgu-knet-baseline: chạy lại baseline 5 của KnowledgeNet (Mesquita et al., EMNLP 2019 — pipeline của chính
paper: spaCy 2.1 + neuralcoref + BERT TF-Hub + bộ phân loại huấn luyện sẵn, liên kết thực thể Diffbot đã cache trong bản
phát hành) trên fold 4 bằng `python evaluate.py dev` của họ. Môi trường 2019 (Python 3.7, TensorFlow 1.13.1) dựng bằng
micromamba (uv không còn bản Python 3.7), ghim numpy/protobuf/gast cho TF 1.13; chạy CPU (TF 1.13 không hỗ trợ CUDA của Kaggle). README của họ: dev span_overlap P 0,718 R 0,691 F1 0,704. Mỗi
bước có timeout; dựng môi trường hỏng thì ghi lỗi và kết thúc (khi đó báo cáo dùng số README).
Đầu ra /kaggle/working/out/: evaluate.log, summary.json.
"""
import hashlib
import json
import os
import re
import subprocess
import time

KN_REPO = "https://github.com/diffbot/knowledge-net.git"
RELEASE = "https://github.com/diffbot/knowledge-net/releases/download/EMNLP2019/knowledge-net-EMNLP2019.zip"
SPACY_LG = ("https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/"
            "en_core_web_lg-2.1.0.tar.gz")
WORK = "/kaggle/working"
OUT, KN = f"{WORK}/out", f"{WORK}/knowledge-net"
BASE = f"{KN}/baselines/EMNLP2019"
VENV = f"{WORK}/venv37"
T0 = time.time()
os.makedirs(OUT, exist_ok=True)
summary: dict = {"steps": {}, "readme_dev_span_overlap": {"P": 0.718, "R": 0.691, "F1": 0.704}}


def step(name: str, cmd: str, timeout_h: float, cwd: str | None = None) -> bool:
    t0 = time.time()
    print(f"$ {cmd}", flush=True)
    try:
        r = subprocess.run(["timeout", f"{timeout_h}h", "bash", "-c", cmd], cwd=cwd, text=True, capture_output=True)
        log = r.stdout + r.stderr
        ok = r.returncode == 0
    except Exception as e:
        log, ok = repr(e), False
    open(f"{OUT}/{name}.log", "w", encoding="utf-8").write(log)
    print(log[-3000:], flush=True)
    summary["steps"][name] = {"ok": ok, "seconds": round(time.time() - t0, 1)}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    return ok


BERT_HUB = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"
HUB_DIR = f"/tmp/tfhub/{hashlib.sha1(BERT_HUB.encode('utf8')).hexdigest()}"      # thư mục cache tensorflow_hub 0.5
PIP = f"{VENV}/bin/python -m pip install -q --no-cache-dir"
CHECK = (f"{VENV}/bin/python -c \"import spacy, neuralcoref, tensorflow; nlp = spacy.load('en_core_web_lg'); "
         f"neuralcoref.add_to_pipe(nlp); print('env ok', tensorflow.__version__)\"")
ok = (step("clone", f"git clone -q --depth 1 {KN_REPO} {KN} && cd {BASE} && wget -q {RELEASE} -O release.zip && "
                    f"unzip -q -o release.zip && ls -la && for d in */; do [ -d $d/models ] || [ -d $d/tmp ] && "
                    f"cp -rn $d. . ; done; cp {KN}/dataset/train.json {KN}/dataset/evaluator.py {KN}/ && "
                    f"ls -la {BASE} {BASE}/models {BASE}/tmp", 0.5)       # config: KNOWLEDGE_NET_DIR="../../", models/, tmp/
      and step("env", f"cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"
                      f" && /tmp/bin/micromamba create -y -q -p {VENV} -c conda-forge python=3.7 pip && "
                      f"{PIP} -r {BASE}/requirements.txt numpy==1.16.6 protobuf==3.20.3 gast==0.2.2 && {PIP} {SPACY_LG} && "
                      f"({CHECK} || ({PIP} --force-reinstall --no-deps --no-binary neuralcoref neuralcoref==4.0 && {CHECK}))",
               1.5)
      and step("bert", f"mkdir -p {HUB_DIR} && curl -sL '{BERT_HUB}?tf-hub-format=compressed' | tar -xz -C {HUB_DIR} && "
                       f"ls {HUB_DIR}", 0.5)
      and step("evaluate", f"cd {BASE} && CUDA_VISIBLE_DEVICES= TFHUB_CACHE_DIR=/tmp/tfhub {VENV}/bin/python evaluate.py dev",
               9))
if os.path.exists(f"{OUT}/evaluate.log"):
    text = open(f"{OUT}/evaluate.log", encoding="utf-8").read()
    for method in ("span_overlap", "span_exact", "uri"):
        block = text.split(f"RESULTS FOR {method}", 1)
        if len(block) == 2 and "Macro Evaluation" in block[1]:      # trước đó là ma trận TP/FP/FN cũng có dòng #global
            m = re.search(r"#global\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", block[1].split("Macro Evaluation", 1)[1])
            if m:
                summary.setdefault("rerun", {})[method] = dict(zip(("P", "R", "F1"), map(float, m.groups())))
summary["ok"] = bool(ok)
summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
subprocess.run(f"rm -rf {KN} {VENV} {BASE}/tmp", shell=True)
print("DONE", flush=True)
