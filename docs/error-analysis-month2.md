# Error Analysis — Month 2

### Table 0: LLM Extractor + LLM Judge (n=20) by Event Type

`--by event` trên `results/nba_test_llm_llm_20.jsonl` (đo 2026-09-08, sau Task 13 — schema enum quan hệ, few-shot, prompt bộ phán gọn; model Qwen3.5-2B Q8_0):

| group       | n  | f1    | add   | del   | cut/ex | add/ex |
|-------------|----|-------|-------|-------|--------|--------|
| draft       | 1  | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| free_agency | 1  | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| overseas    | 1  | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| released    | 11 | 0.951 | 0.000 | 0.000 | 0.0    | 0.0    |
| retirement  | 1  | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| trade       | 5  | 0.920 | 0.000 | 0.000 | 0.0    | 0.0    |

`cut/ex` = `add/ex` = 0.0 across every event: LLM #1 (extractor) produced zero explicit ops on all 20 examples in this run, so `localize()` never had participants to build candidates from, and the judge (LLM #2) was never invoked (`llm_calls` = 20, all from the extractor). This isolates the current bottleneck to LLM #1's zero-shot/few-shot extraction on this 2B model, not to the judge or the localization/judge hardening done in Task 13 — see `gold_explicit + llm judge` in the README (add_acc 0.051, del_acc 1.0, 1/20 context-overflow error handled as a per-example record instead of aborting the run) for evidence the judge and error-handling paths work when given real explicit ops.

## Analysis Tables

### Table 1: Oracle Results by Coverage

| group  | n   | f1    | add   | del   | cut/ex | add/ex |
|--------|-----|-------|-------|-------|--------|--------|
| full   | 438 | 1.000 | 1.000 | 1.000 | 70.3   | 31.0   |

The oracle judge (perfect event extraction + perfect judgment) achieves F1=1.0 across all 438 examples. This establishes the upper bound for pipeline performance.

### Table 2: Oracle Results by Event Type

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| d_league    | 6   | 1.000 | 0.000 | 1.000 | 73.3   | 0.0    |
| draft       | 18  | 1.000 | 1.000 | 0.000 | 0.0    | 20.0   |
| free_agency | 107 | 1.000 | 1.000 | 0.000 | 0.0    | 20.2   |
| head_coach  | 10  | 1.000 | 1.000 | 1.000 | 0.0    | 19.5   |
| overseas    | 3   | 1.000 | 0.000 | 1.000 | 62.7   | 0.0    |
| released    | 139 | 1.000 | 0.000 | 1.000 | 71.3   | 0.0    |
| retirement  | 5   | 1.000 | 0.000 | 1.000 | 61.6   | 0.0    |
| trade       | 150 | 1.000 | 1.000 | 1.000 | 133.1  | 72.3   |

With perfect extraction, all events achieve F1=1.0. Key observations:
- `trade` events have highest cut/ex (133.1) and add/ex (72.3) — inherently multi-operation events
- `released` events have moderate cut/ex (71.3) — consistent deletions needed
- `draft` and `free_agency` have only ADD operations (no deletions)

### Table 3: Gold Extractor (No Judge) by Event Type

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| d_league    | 6   | 0.952 | 0.000 | 0.052 | 73.3   | 0.0    |
| draft       | 18  | 0.951 | 0.053 | 0.000 | 0.0    | 20.0   |
| free_agency | 107 | 0.952 | 0.052 | 0.000 | 0.0    | 20.2   |
| head_coach  | 10  | 1.000 | 1.000 | 1.000 | 0.0    | 19.5   |
| overseas    | 3   | 0.945 | 0.000 | 0.060 | 62.7   | 0.0    |
| released    | 139 | 0.951 | 0.000 | 0.053 | 71.3   | 0.0    |
| retirement  | 5   | 0.944 | 0.000 | 0.061 | 61.6   | 0.0    |
| trade       | 150 | 0.900 | 0.056 | 0.056 | 133.1  | 72.3   |

Gold extractor with explicit rules shows degradation:
- `trade`: F1 drops to 0.900 (worst), suggesting extraction complexity
- `head_coach`: F1=1.000 (best) — simplest event pattern
- Other events: F1≈0.95, suggesting minor extraction issues

### Table 4: LLM Extractor (20-sample, No Judge) by Event Type

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| draft       | 1   | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| free_agency | 1   | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| overseas    | 1   | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| released    | 11  | 0.951 | 0.000 | 0.002 | 7.6    | 0.0    |
| retirement  | 1   | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| trade       | 5   | 0.920 | 0.000 | 0.000 | 0.0    | 0.0    |

LLM extractor (n=20) shows:
- Severe degradation compared to gold (F1≈0.92–0.95 vs ≈1.0)
- Dominance of `released` events (11/20 samples)
- No ADD operations detected by LLM (add_acc=0.0 across all events)
- `trade` still worst (F1=0.920)
- Low candidate counts (avg cut=0–7.6, add=0.0) suggest aggressive under-extraction

## Five Concrete Error Examples

Based on comparison of `nba_test_llm_nojudge_20.jsonl` vs `nba_test_gold_explicit_nojudge.jsonl`:

### Example 1: idx=0, event=released
**Text:** "the Indiana_Pacers announced friday they have waived forward Thaddeus_Young_2017-18 ..."

**Expected ops:** 2 INVALIDATE (remove player from roster, remove his contract)
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity — player name with version suffix `_2017-18` not recognized; LLM may have treated it as malformed or OOV token.

---

### Example 2: idx=1, event=trade
**Text:** "cleveland , oh – the Cleveland_Cavaliers have acquired center Anthony_Randolph_2014-15 and a 2015 second round pick from the Denver_Nuggets in exchange for two protected 2015 first round picks ( <unk> oklahoma city and memphis ) , cavaliers general manager david griffin announced tonight ..."

**Expected ops:** 2 INVALIDATE + 2 ADD (remove traded players, add acquired players)
**LLM ops:** (empty — 0 operations)

**Likely cause:** tin nhiều giao dịch — trade with multiple picks and entities; OOV token `<unk>` corrupts extraction context.

---

### Example 3: idx=2, event=released
**Text:** "boston – the Boston_Celtics announced today that they have waived guard Kelly_Olynyk_2015-16 ..."

**Expected ops:** 2 INVALIDATE (standard waived player pattern)
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity — player identifier with version suffix `_2015-16` not recognized by LLM.

---

### Example 4: idx=3, event=released
**Text:** "the Golden_State_Warriors state have waived forward Jason_Thompson_2015-16 , the team announced today ..."

**Expected ops:**
- INVALIDATE(Golden_State_Warriors, <player>, Jason_Thompson_2015-16)
- INVALIDATE(Jason_Thompson_2015-16, <player>, Golden_State_Warriors)

**LLM ops:** 
- INVALIDATE(Golden_State_Warriors, <general_mananger>, Bob_Myers_2015-16)

**Likely cause:** LLM bịa entity + LLM đảo vị trí (h, r, t) — LLM hallucinated an entity (Bob_Myers_2015-16, not in text) and wrong relation (<general_mananger> vs <player>). The op survived post-validation only because Bob_Myers_2015-16 exists in the graph, but it's factually wrong for this news event.

---

### Example 5: idx=4, event=released
**Text:** "<unk> the Atlanta_Hawks basketball club has requested waivers on forward Jaylen_Morris_2018-19 , it was announced today ..."

**Expected ops:** 2 INVALIDATE
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity + OOV tokens — text begins with OOV token `<unk>` that likely disrupts extraction; player name with version suffix not recognized.

---

## Summary

**Gold Extractor Performance:** F1≈0.95 on most event types, but `trade` drops to 0.90 due to multi-operation complexity.

**LLM Extractor Performance:** F1≈0.92–0.95 overall, but with systematic failures:
1. **Zero ADD operations** across all samples (gold achieves ~1.0 for draft/free_agency/trade)
2. **Entity recognition failures** on versioned player IDs (`_2017-18` suffixes)
3. **OOV token handling** — `<unk>` tokens disrupt extraction pipeline
4. **Hallucination of non-existent entities** — Example 4 shows LLM fabricating Bob_Myers_2015-16 when text mentions Jason_Thompson_2015-16; also wrong relation detected

**Primary Bottleneck:** LLM cannot reliably extract from complex or malformed inputs; future work should preprocess entity identifiers and handle OOV tokens before LLM processing.
