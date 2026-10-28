"""Kernel kgu-knet-baseline: chạy lại baseline 5 của KnowledgeNet (Mesquita et al., EMNLP 2019 — pipeline của chính
paper: spaCy 2.1 + neuralcoref + BERT TF-Hub + bộ phân loại huấn luyện sẵn, liên kết thực thể Diffbot đã cache trong bản
phát hành) trên fold 4 bằng `python evaluate.py dev` của họ. Môi trường 2019 (Python 3.7, TensorFlow 1.13.1) dựng bằng
micromamba (uv không còn bản Python 3.7), ghim numpy/protobuf/gast cho TF 1.13; chạy CPU (TF 1.13 không hỗ trợ CUDA của Kaggle). README của họ: dev span_overlap P 0,718 R 0,691 F1 0,704. Mỗi
bước có timeout; dựng môi trường hỏng thì ghi lỗi và kết thúc (khi đó báo cáo dùng số README).
Lần chạy thẳng `evaluate.py dev` (v4) chết lặng lẽ ở văn bản 282/1.022 (không traceback: hết RAM hoặc neuralcoref
crash) → dự đoán bằng chính run.generate_candidates / run.classify_instances như evaluate.py nhưng theo khúc 100 văn
bản, mỗi khúc một tiến trình (KN_PREDICT dưới đây), ghi từng văn bản xong ra predictions.jsonl; tiến trình chết giữa
khúc thì bỏ đúng văn bản đang chạy (ghi vào summary "failed_docs", fact của nó tính là bỏ sót) và chạy tiếp. Chấm bằng
dataset/evaluator.py của tác giả, cùng cờ với scripts/score_knet.py (-c -f 4).
Đầu ra /kaggle/working/out/: predictions.jsonl, pred_baseline5.json, eval_<cách chấm>.txt, predict_*.log, summary.json.
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
PIP = f"{VENV}/bin/python -m pip install --no-cache-dir --prefer-binary --no-build-isolation"     # sdist 2019 cần setuptools cũ
CHECK = (f"{VENV}/bin/python -c \"import spacy, neuralcoref, tensorflow; nlp = spacy.load('en_core_web_lg'); "
         f"neuralcoref.add_to_pipe(nlp); print('env ok', tensorflow.__version__)\"")
KN_PREDICT = r"""# dự đoán như evaluate.py dev (fold 4) cho văn bản thứ start..end-1 (thứ tự trong train.json), mỗi văn bản một dòng
import json, os, sys
import run, evaluator
from config import KNOWLEDGE_NET_DIR
start, end, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
order = [json.loads(x)["documentId"] for x in open(os.path.join(KNOWLEDGE_NET_DIR, "train.json")) if x.strip()]
gold, _ = evaluator.readKnowledgenetFile(os.path.join(KNOWLEDGE_NET_DIR, "train.json"), 4)
order = [d for d in order if d in gold]
with open(out, "a") as f:
  for i in range(start, min(end, len(order))):
    document = gold[order[i]]
    print("DOC", i, document.documentId, flush=True)
    instances = run.generate_candidates(document.documentText)
    passages = []
    for passage in document.passages:
      annotated = set(map(lambda x: x.propertyId, passage.exhaustivelyAnnotatedProperties))
      if len(annotated) == 0:
        continue
      pi = list(filter(lambda x: x.is_in_span(passage.passageStart, passage.passageEnd), instances))
      run.classify_instances(pi, annotated)
      facts = []
      for fact in pi:
        for pid, label in fact.labels.items():
          if label:
            facts.append({"propertyId": pid, "subjectStart": fact.subject_entity.start_char,
                          "subjectEnd": fact.subject_entity.end_char, "objectStart": fact.object_entity.start_char,
                          "objectEnd": fact.object_entity.end_char, "subjectUri": fact.get_subject_uri(pid) or "",
                          "objectUri": fact.get_object_uri(pid) or "", "subjectText": str(fact.subject_entity),
                          "objectText": str(fact.object_entity)})
      passages.append({"passageId": passage.passageId, "facts": facts})
    f.write(json.dumps({"i": i, "documentId": document.documentId, "passages": passages}) + "\n")
    f.flush()
"""
ok = (step("clone", f"git clone -q --depth 1 {KN_REPO} {KN} && cd {BASE} && wget -q {RELEASE} -O release.zip && "
                    f"unzip -q -o release.zip && find knowledge-net-EMNLP2019 -maxdepth 5 | head -80 && "
                    f"for n in models tmp vocab; do d=$(find knowledge-net-EMNLP2019 -type d -name $n | head -1); "
                    f"[ -n \"$d\" ] && cp -rn $d . ; done; cp {KN}/dataset/train.json {KN}/dataset/evaluator.py {KN}/ && "
                    f"ls -la {BASE} {BASE}/models {BASE}/tmp", 0.5)       # config: KNOWLEDGE_NET_DIR="../../", models/, tmp/
      and step("env", f"cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"
                      f" && /tmp/bin/micromamba create -y -q -p {VENV} -c conda-forge python=3.7 pip && "
                      f"{PIP} 'setuptools<58' wheel 'cython<3' numpy==1.16.6 && "
                      f"{PIP} -r {BASE}/requirements.txt numpy==1.16.6 protobuf==3.20.3 gast==0.2.2 && {PIP} {SPACY_LG} && "
                      f"({CHECK} || ({PIP} --force-reinstall --no-deps --no-binary neuralcoref neuralcoref==4.0 && {CHECK}))",
               1.5)
      and step("bert", f"mkdir -p {HUB_DIR} && curl -sL '{BERT_HUB}?tf-hub-format=compressed' | tar -xz -C {HUB_DIR} && "
                       f"ls {HUB_DIR}", 0.5))
if ok:
    open(f"{BASE}/kn_predict.py", "w", encoding="utf-8").write(KN_PREDICT)
CHUNK = 100
failed: list[int] = []
if ok:
    gold = [json.loads(x) for x in open(f"{KN}/dataset/train.json", encoding="utf-8") if x.strip()]
    gold = [d for d in gold if d["fold"] == 4]
    n, pos, pred_file = len(gold), 0, f"{OUT}/predictions.jsonl"
    summary["n_docs"] = n

    def done_idx() -> set[int]:
        if not os.path.exists(pred_file):
            return set()
        return {json.loads(x)["i"] for x in open(pred_file, encoding="utf-8") if x.strip()}

    while pos < n and time.time() - T0 < 10.5 * 3600:
        end = min(pos + CHUNK, n)
        step(f"predict_{pos:04d}", f"cd {BASE} && CUDA_VISIBLE_DEVICES= TFHUB_CACHE_DIR=/tmp/tfhub {VENV}/bin/python "
                                   f"kn_predict.py {pos} {end} {pred_file}", 3)
        got = [i for i in done_idx() if pos <= i < end]
        nxt = max(got) + 1 if got else pos
        if nxt < end:                     # chết giữa khúc → bỏ văn bản đang chạy
            failed.append(nxt)
            nxt += 1
        pos = nxt
        summary["failed_docs"], summary["done_docs"] = failed, len(done_idx())
    by_doc = {}
    for x in open(pred_file, encoding="utf-8") if os.path.exists(pred_file) else []:
        r = json.loads(x)
        by_doc[str(r["documentId"])] = {p["passageId"]: p["facts"] for p in r["passages"]}
    with open(f"{OUT}/pred_baseline5.json", "w", encoding="utf-8") as f:
        for d in gold:
            facts = by_doc.get(str(d["documentId"]), {})
            f.write(json.dumps({**d, "passages": [{**p, "facts": facts.get(p["passageId"], [])}
                                                   for p in d["passages"]]}, ensure_ascii=False) + "\n")
    for method in ("span_overlap", "span_exact", "uri"):
        r = subprocess.run(["python", f"{KN}/dataset/evaluator.py", f"{KN}/dataset/train.json",
                            f"{OUT}/pred_baseline5.json", "-e", method, "-c", "-f", "4"],
                           capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
        open(f"{OUT}/eval_{method}.txt", "w", encoding="utf-8").write(r.stdout + r.stderr)
        m = re.search(r"#global\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", r.stdout.split("Macro Evaluation", 1)[-1])
        if m:
            summary.setdefault("rerun", {})[method] = dict(zip(("P", "R", "F1"), map(float, m.groups())))
    ok = "rerun" in summary
summary["ok"] = bool(ok)
summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
subprocess.run(f"rm -rf {KN} {VENV} {BASE}/tmp", shell=True)
print("DONE", flush=True)
