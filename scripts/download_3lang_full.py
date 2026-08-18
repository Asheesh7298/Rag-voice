"""
Download full 3-language dataset using HuggingFace datasets streaming.
This streams row-by-row without downloading entire multi-GB parquet files.

Run: python scripts/download_3lang_full.py
"""
import os
import sys
import json
import time
import hashlib

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUT_PATH = os.path.join(PROCESSED_DIR, "passages_3lang_full.jsonl")


def stream_msmarco_xi_lang(lang, out_f, stats):
    """Stream a single language from MSMARCO-XI using HuggingFace datasets."""
    from datasets import load_dataset
    
    print(f"\n[{lang.upper()}] Streaming from ai4bharat/MSMARCO-XI...")
    t0 = time.perf_counter()
    
    try:
        # MSMARCO-XI has one default config with source_lang/target_lang columns
        ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading MSMARCO-XI: {e}")
        return
    
    # Map language code to MSMARCO-XI target language code
    lang_map = {"hi": "hin", "mr": "mar"}
    target_code = lang_map.get(lang)
    
    lang_written = 0
    seen_texts = set()
    
    for row in ds:
        # Filter by target language
        row_target = row.get("target_lang", "")
        if row_target != target_code:
            continue
        
        target_text = str(row.get("target_text", "")).strip()
        source_text = str(row.get("source_text", "")).strip()
        
        if not target_text or len(target_text) < 20:
            continue
        
        text_hash = hashlib.md5(target_text.encode("utf-8")).hexdigest()[:12]
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        
        query_id = str(row.get("query_id", lang_written))
        query = str(row.get("query", "")).strip()
        answers = row.get("answers", [])
        
        if isinstance(answers, list) and answers:
            answer = str(answers[0]).strip()
        else:
            answer = ""
        
        record = {
            "id": f"{lang}-{query_id}-p{lang_written}",
            "lang": lang,
            "query_id": f"{lang}-{query_id}",
            "query": query,
            "text": target_text,
            "is_selected": True,
            "answers": [answer] if answer else [],
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        lang_written += 1
        
        if lang_written % 50000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{lang.upper()}] {lang_written:,} passages ({elapsed:.0f}s)...")
    
    stats[lang] = lang_written
    elapsed = time.perf_counter() - t0
    print(f"  [{lang.upper()}] Done: {lang_written:,} passages in {elapsed:.0f}s")


def download_indicmsmarco_lang(lang, out_f, stats):
    """Download from IndicMSMARCO (1000 rows/lang) as a reliable fallback."""
    from datasets import load_dataset
    
    print(f"\n[{lang.upper()}] Loading from ai4bharat/IndicMSMARCO...")
    t0 = time.perf_counter()
    
    try:
        ds = load_dataset("ai4bharat/IndicMSMARCO", lang, split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    lang_written = 0
    seen_texts = set()
    
    for i, row in enumerate(ds):
        text = (row.get("passage") or "").strip()
        if not text or len(text) < 20:
            continue
        
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        
        query_id = row.get("query_id") or str(i)
        passage_id = row.get("passage_id") or str(i)
        query = row.get("query", "")
        answer = row.get("answer", "")
        
        record = {
            "id": f"{lang}-{query_id}-p{passage_id}",
            "lang": lang,
            "query_id": f"{lang}-{query_id}",
            "query": query,
            "text": text,
            "is_selected": bool(row.get("is_selected", False)),
            "answers": [answer] if answer else [],
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        lang_written += 1
    
    stats[lang] = lang_written
    elapsed = time.perf_counter() - t0
    print(f"  [{lang.upper()}] Done: {lang_written:,} passages in {elapsed:.0f}s")


def download_english_msmarco(out_f, stats):
    """Download English from original MS MARCO v2.1 via streaming."""
    from datasets import load_dataset
    
    print("\n[EN] Streaming English from ms_marco v2.1...")
    t0 = time.perf_counter()
    
    try:
        ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Falling back to IndicMSMARCO passages as English source...")
        download_indicmsmarco_as_english(out_f, stats)
        return
    
    en_written = 0
    seen_texts = set()
    
    for row in ds:
        query = str(row.get("query", "")).strip()
        query_id = str(row.get("query_id", en_written))
        answers = row.get("answers", [])
        passages = row.get("passages", {})
        
        if isinstance(passages, dict):
            passage_texts = passages.get("passage_text", [])
            is_selected_list = passages.get("is_selected", [])
        else:
            continue
        
        if not passage_texts:
            continue
        
        if isinstance(answers, list) and answers:
            answer = str(answers[0]).strip()
            if answer.lower() == "no answer present.":
                answer = ""
        else:
            answer = ""
        
        for p_idx, p_text in enumerate(passage_texts):
            text = str(p_text).strip()
            if not text or len(text) < 20:
                continue
            
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            
            is_sel = bool(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else False
            
            record = {
                "id": f"en-{query_id}-p{p_idx}",
                "lang": "en",
                "query_id": f"en-{query_id}",
                "query": query,
                "text": text,
                "is_selected": is_sel,
                "answers": [answer] if answer else [],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            en_written += 1
        
        if en_written % 100000 == 0 and en_written > 0:
            elapsed = time.perf_counter() - t0
            print(f"  [EN] {en_written:,} passages ({elapsed:.0f}s)...")
    
    stats["en"] = en_written
    elapsed = time.perf_counter() - t0
    print(f"  [EN] Done: {en_written:,} passages in {elapsed:.0f}s")


def download_indicmsmarco_as_english(out_f, stats):
    """Fallback: use IndicMSMARCO Hindi passages as bilingual English source."""
    from datasets import load_dataset
    
    en_written = 0
    for src_lang in ["hi", "mr"]:
        try:
            ds = load_dataset("ai4bharat/IndicMSMARCO", src_lang, split="train")
            for i, row in enumerate(ds):
                text = (row.get("passage") or "").strip()
                if not text:
                    continue
                record = {
                    "id": f"en-from-{src_lang}-{i}",
                    "lang": "en",
                    "query_id": f"en-{src_lang}-{i}",
                    "query": row.get("query", ""),
                    "text": text,
                    "is_selected": bool(row.get("is_selected", False)),
                    "answers": [row.get("answer", "")] if row.get("answer") else [],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                en_written += 1
        except Exception:
            pass
    stats["en"] = en_written
    print(f"  [EN FALLBACK] Wrote {en_written:,} passages")


def main():
    print("=" * 80)
    print("FULL 3-LANGUAGE DATASET DOWNLOAD (HINDI + MARATHI + ENGLISH)")
    print("=" * 80)
    
    stats = {}
    t_start = time.perf_counter()
    
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        # 1. Hindi and Marathi from IndicMSMARCO (guaranteed 1000 rows each, fast)
        for lang in ["hi", "mr"]:
            download_indicmsmarco_lang(lang, out_f, stats)
        
        # 2. English from MS MARCO v2.1 (streaming, millions of passages)
        download_english_msmarco(out_f, stats)
        
        # 3. Try to supplement with MSMARCO-XI streaming for more Hindi/Marathi
        print("\n--- Supplementing with MSMARCO-XI translations ---")
        try:
            from datasets import load_dataset
            ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
            
            hi_extra = 0
            mr_extra = 0
            seen = set()
            
            for row in ds:
                target_lang = row.get("target_lang", "")
                target_text = str(row.get("target_text", "")).strip()
                
                if target_lang not in ("hin", "mar"):
                    continue
                if not target_text or len(target_text) < 20:
                    continue
                
                text_hash = hashlib.md5(target_text.encode("utf-8")).hexdigest()[:12]
                if text_hash in seen:
                    continue
                seen.add(text_hash)
                
                lang_code = "hi" if target_lang == "hin" else "mr"
                query_id = str(row.get("query_id", ""))
                
                record = {
                    "id": f"{lang_code}-xi-{query_id}-{hi_extra + mr_extra}",
                    "lang": lang_code,
                    "query_id": f"{lang_code}-xi-{query_id}",
                    "query": str(row.get("query", "")),
                    "text": target_text,
                    "is_selected": True,
                    "answers": [],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if lang_code == "hi":
                    hi_extra += 1
                else:
                    mr_extra += 1
                
                total_extra = hi_extra + mr_extra
                if total_extra % 100000 == 0 and total_extra > 0:
                    print(f"  [MSMARCO-XI] hi={hi_extra:,} mr={mr_extra:,}")
            
            stats["hi"] = stats.get("hi", 0) + hi_extra
            stats["mr"] = stats.get("mr", 0) + mr_extra
            print(f"  [MSMARCO-XI] Added hi={hi_extra:,} mr={mr_extra:,}")
            
        except Exception as e:
            print(f"  [MSMARCO-XI] Skipping supplemental: {e}")
    
    total = sum(stats.values())
    elapsed = time.perf_counter() - t_start
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    for lang, count in sorted(stats.items()):
        print(f"  {lang.upper()}: {count:,} passages")
    print(f"  TOTAL: {total:,} passages")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Output: {OUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
