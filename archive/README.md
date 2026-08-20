# Archive: Experimental, Scratch & Diagnostic Scripts

This directory contains utility scripts, one-off patches, diagnostic tools, and exploration experiments developed during the lifecycle of the Voice RAG project. They are archived here to keep the root directory and production structure clean.

---

## 📂 Category Breakdown & Explanations

### 1. 🧪 Patch & Optimization Scripts (`archive/patch_*.py`)
These scripts tested isolated pipeline improvements before integrating them into `modal_app.py`:
- `patch_modal.py`: Early patch script for testing container CPU and GPU memory parameters on Modal.
- `patch_warmup.py`: Validated zero-latency cold-start warmup for FAISS and transformer models during container boot.
- `patch_top4_qa.py`: Tested top-4 passage concatenation vs top-1 passage extractive QA.
- `patch_span_matrix.py` & `patch_span_filter.py`: Evaluated multi-passage start/end logit matrix aggregation for extractive span boundary accuracy.
- `patch_rank_decay.py`: Experimented with rank-based score decay penalties for lower-ranked retrieval candidates.
- `patch_retrieval_morphology.py`: Explored character n-gram morphology matching for Devanagari inflections.
- `patch_lang_isolation.py` & `patch_lang_isolation_v2.py`: Prototyped strict language filtering to prevent cross-language retrieval leakage.
- `patch_guardrails_v2.py`: Tested threshold adjustments for Guardrails 1–7.
- `patch_intent_qa.py`: Experimented with query intent classification before extractive QA.
- `patch_expand_sentence.py` & `patch_concise.py`: Tested sentence expansion around extracted spans for improved answer fluency.
- `patch_facts_v2.py`, `patch_core_facts.py`, `remove_core_facts.py`: Explored synthetic core-fact extraction vs direct raw passage retrieval.
- `patch_test_display.py`: Prototyped formatted Devanagari output rendering in terminal.

---

### 2. 🔬 Scratch & Prototyping Scripts (`archive/scratch_*.py`)
Used for rapid hypothesis testing and deep-dive inspections:
- `scratch_eval.py`: Early multi-metric evaluation harness.
- `scratch_inspect_dataset.py`: Inspected raw MS MARCO Indic and English parquet structures.
- `scratch_check_span_positions.py`: Verified ground-truth answer character offsets within source passages.
- `scratch_test_batched_qa.py`: Measured batch-size throughput of XLM-RoBERTa on A10G GPU.
- `scratch_test_qa_termite.py`: Debugged span extraction for technical biological queries.
- `scratch_test_span_matrix.py`: Unit test for logit span decoding algorithm.
- `scratch_trace_decline.py`: Traced exact guardrail trigger points on failing queries.
- `scratch_train_qa.py`: Evaluated feasibility of fine-tuning QA head vs pre-trained `xlm-roberta-base-squad2`.

---

### 3. 🔍 Dataset & Index Diagnostics (`archive/check_*.py`, `archive/inspect_*.py`)
Used to verify data scale, balance, and volume integrity:
- `check_3lang_counts.py`: Verified 500k Hindi + 500k Marathi + 500k English passage distribution.
- `check_capitals.py`: Verified capital city question coverage across Indic languages.
- `check_corpus_queries.py`: Inspected query-passage alignment in MS MARCO splits.
- `check_hf_file_sizes.py`: Checked Hugging Face dataset download chunk sizes.
- `check_splits.py`: Validated train/dev/test partition counts.
- `inspect_hf_repo.py`: Explored `ai4bharat/MSMARCO-XI` dataset repository structures.
- `inspect_min_score.py`: Inspected distribution of cosine similarity retrieval scores across languages.
- `inspect_failing_queries.py` & `inspect_failing_50.py`: Isolated queries triggering false declines.
- `diagnose_40.py`: Deep diagnostic on 40-query test set.

---

### 4. 🤖 Model & SLM Exploration (`archive/test_*.py`, `archive/list_*.py`)
Explored alternative LLM / SLM architectures (Qwen, Gemini Flash) before settling on high-speed GPU extractive QA:
- `test_qwen_slm.py`: Evaluated Qwen-2.5-0.5B-Instruct generation latency (found extractive QA was 4x faster).
- `test_slm_options.py`: Benchmarked small language model options for sub-200ms latency.
- `test_gemini_rag.py`, `test_gemini_models.py`, `test_gemini_models_v2.py`, `test_gemini_key.py`: Tested Google Gemini Flash API as generation backend.
- `list_models.py`: Queried available Hugging Face and Gemini models.
- `test_6_queries.py`, `test_all_6.py`, `test_raw_query.py`, `test_extra.py`, `test_factual_examples.py`, `test_new_deployment.py`: Rapid sanity-check scripts for live endpoints.

---

### 5. 📜 Legacy Benchmark Scripts (`archive/scripts_legacy/`)
Earlier benchmark iterations replaced by the official 90-question benchmark suite:
- `benchmark_40.py`, `benchmark_50_loop.py`, `compare_40_benchmarks.py`: Earlier 40/50 query benchmarks.
- `generate_50_eval.py`, `generate_pure_3lang_120.py`, `generate_real_120_benchmark.py`: Synthetic test set generators.
- `detailed_eval_26.py`, `eval_all_metrics.py`, `debug_declines.py`: Older evaluation scripts.
- `complete_90.py`, `extract_more_mr_candidates.py`, `finalize_90_set2.py`, `find_clean_examples.py`, `generate_90_verified_benchmark.py`, `verify_and_build_90_bench.py`: Intermediate data preparation scripts used to construct the official 90-question benchmark suites (`benchmark_90.py` and `benchmark_90_set2.py`).
- `multilang_test.py`: Multi-language prototype test.
- `rebuild_index.py`, `modal_index_2_76m.py`: Previous 2.76M index build attempt before optimizing to the 1.51M multi-strategy target scale.
- `benchmark_1000_results.md`: Early 1,000-query benchmark run on the initial 58k-vector T4 GPU prototype.
