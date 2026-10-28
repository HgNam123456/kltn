"""Vá EDC (clear-nus/edc @02aad17) để chạy với llama-server, dùng chung cho các kernel benchmark (như
kaggle/atom_edc/run_atom_edc.py, nơi các vá này được kiểm lần đầu):
- embedder nạp fp16 trên $EDC_EMB_DEVICE (bản gốc fp32 = 28 GB);
- nhánh OpenAI không giải phóng verify model chưa từng nạp (lỗi gốc);
- dòng log debug của canonicalization đọc biến vòng lặp chưa gán khi câu không có triple OIE (lỗi gốc);
- EDC.oie đọc triple có sẵn từ $EDC_OIE_FROM (json: list triple mỗi dòng input) thay cho bước OIE của EDC — để chạy
  bước Schema Definition + Canonicalization của EDC trên triple của hệ khác (ATOM, KGGen);
- import e5_mistral_utils (chỉ cho refinement, nạp huggingface_hub.Repository đã bị gỡ) bọc try/except.

Dùng: python scripts/edc_patch.py <thư mục repo edc>
"""
from __future__ import annotations

import sys
from pathlib import Path

IMPORT = "from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding"


def _replace(src: str, old: str, new: str, what: str) -> str:
    assert src.count(old) == 1, f"vá {what} không khớp source"
    return src.replace(old, new)


def patch(edc_dir: str | Path) -> None:
    fw = Path(edc_dir) / "edc" / "edc_framework.py"
    src = fw.read_text(encoding="utf-8")
    src = _replace(src, "model = SentenceTransformer(model_name, trust_remote_code=True)",
                   "model = SentenceTransformer(model_name, trust_remote_code=True, device=os.environ['EDC_EMB_DEVICE'], "
                   "model_kwargs={'torch_dtype': __import__('torch').float16})", "embedder")
    src = _replace(src, "            llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
                        "            del self.loaded_model_dict[self.sc_llm_name]\n",
                   "            if not llm_utils.is_model_openai(self.sc_llm_name):\n"
                   "                llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)\n"
                   "                del self.loaded_model_dict[self.sc_llm_name]\n", "free_model")
    src = _replace(src, 'logger.debug(f"Retrieved candidate relations {canon_candidate_dict}")',
                   'logger.debug(f"Retrieved candidate relations {canon_candidate_dict_list}")', "log")
    old = "previous_extracted_triplets_list: List[List[str]] = None, free_model=False\n    ):\n"
    src = _replace(src, old, old + (
        "        if os.environ.get('EDC_OIE_FROM'):  # vá kgu: triple OIE có sẵn thay bước OIE của EDC\n"
        "            oie_triples_list = json.load(open(os.environ['EDC_OIE_FROM'], encoding='utf-8'))\n"
        "            assert len(oie_triples_list) == len(input_text_list), (len(oie_triples_list), len(input_text_list))\n"
        "            return oie_triples_list, ['' for _ in input_text_list], ['' for _ in input_text_list]\n"), "oie")
    fw.write_text(src, encoding="utf-8")
    for f in sorted((Path(edc_dir) / "edc").glob("*.py")):
        s = f.read_text(encoding="utf-8")
        if IMPORT in s:
            f.write_text(s.replace(IMPORT, f"try:\n    {IMPORT}\nexcept Exception:  # chỉ dùng cho refinement có adapter\n"
                                           "    MistralForSequenceEmbedding = None"), encoding="utf-8")


if __name__ == "__main__":
    patch(sys.argv[1])
    print("EDC đã vá:", sys.argv[1])
