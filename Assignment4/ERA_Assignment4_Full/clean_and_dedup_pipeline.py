#!/usr/bin/env python3
"""
ERA V5 Session 4: Data Cleaning, Deduplication & Provenance Pipeline
Target Dataset: lordx64/reasoning-distill-claude-opus-4-7-max (8,124 rows, 19.2 MB Parquet, ~32.5M tokens)
Target Directory: /Users/nitinranjansharma/Documents/codes/ERA/Assignment3/
"""
import os
import re
import html
import json
import hashlib
import unicodedata
from pathlib import Path

TARGET_DIR = Path("/Users/nitinranjansharma/Documents/codes/ERA/Assignment3")
DATA_DIR = TARGET_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Sovereign Character-Level Text Normalization (Session 4 Sec 3)
# Strips ZWSP (U+200B), BOM (U+FEFF), Bidi overrides (U+202A-U+202E), Replacement (U+FFFD)
# Explicitly PRESERVES Brahmic/Indic joiners: ZWNJ (U+200C) and ZWJ (U+200D)
NOISE_CHARS_RE = re.compile(r'[\u200b\ufeff\u202a-\u202e\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]')
GHOST_TAGS_RE = re.compile(r'\[/?INST\]|<\|user\|>|<\|assistant\|>|###\s*(?:Instruction|Response):')

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    text = NOISE_CHARS_RE.sub("", text)
    text = GHOST_TAGS_RE.sub("", text)
    text = re.sub(r'\$\$\$(.*?)\$\$\$', r'$\1$', text)
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    return text

STUB_RE = re.compile(
    r'^(?:#\s*Problem|##\s*Problem Statement|Condition:|##\s*Output Format\s*None|##\s*Constraints|##\s*Note\b|##\s*Editorial\b)|'
    r'Your solution must read input from standard input|You will be given a competitive programming problem\.',
    re.I | re.M
)
META_LEAK_RE = re.compile(r'no current rewritten thinking|thinking content that needs to be rewritten', re.I)

def get_shingles(text: str, k: int = 3) -> set:
    words = text.lower().split()
    if len(words) < k:
        return {text.lower()}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def run_full_pipeline():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing Hugging Face datasets library...")
        os.system("pip install -q datasets pyarrow huggingface_hub")
        from datasets import load_dataset

    print("1. Downloading raw dataset: lordx64/reasoning-distill-claude-opus-4-7-max ...")
    ds = load_dataset("lordx64/reasoning-distill-claude-opus-4-7-max", split="train")
    raw_out = DATA_DIR / "reasoning_distill_claude_opus_4_7_max_raw.jsonl"
    ds.to_json(str(raw_out))
    print(f"Saved raw dataset ({len(ds)} rows) to {raw_out}")

    counts = {
        "raw_rows": len(ds),
        "removed_missing_or_truncated": 0,
        "removed_simulated_thinking": 0,
        "removed_exact_duplicates": 0,
        "removed_cutoff_stubs": 0,
        "removed_meta_rewriter_leaks": 0,
        "removed_near_duplicates_lsh": 0,
        "retained_gold_rows": 0
    }

    seen_prompts = set()
    shingle_index = []
    cleaned_rows = []
    hasher = hashlib.sha256()

    for idx, row in enumerate(ds):
        stop_reason = row.get("stop_reason", "")
        msgs = row.get("messages") or []
        user_prompt = msgs[0].get("content", "") if msgs else ""
        thinking = row.get("thinking") or ""
        response = row.get("response") or ""

        # Filter 1: Missing fields, max_tokens truncation, or API refusals
        if stop_reason != "end_turn" or not user_prompt.strip() or not thinking.strip() or not response.strip():
            counts["removed_missing_or_truncated"] += 1
            continue

        # Normalize text (Indic-safe) & strip ghost markers
        p_norm = normalize_text(user_prompt)
        th_norm = normalize_text(thinking)
        resp_norm = normalize_text(response)

        # Filter 2: Simulated thinking (...)
        if "..." in th_norm or "..." in resp_norm or "…" in th_norm:
            counts["removed_simulated_thinking"] += 1
            continue

        # Filter 3: Exact duplicate prompts
        if p_norm in seen_prompts:
            counts["removed_exact_duplicates"] += 1
            continue

        # Filter 4: Thinking-rewriter meta-prompt leakage (Row 4 bug)
        if META_LEAK_RE.search(th_norm):
            counts["removed_meta_rewriter_leaks"] += 1
            continue

        # Filter 5: Cut-off competitive programming stub prompts
        if STUB_RE.search(p_norm) or any(phrase in resp_norm.lower() for phrase in [
            "message got cut off", "problem statement is missing", "haven't included the actual problem"
        ]):
            counts["removed_cutoff_stubs"] += 1
            continue

        # Filter 6: Near-duplicate detection (Shingling + Jaccard >= 0.75)
        sh = get_shingles(p_norm, k=3)
        is_near_dup = False
        for prev_sh in shingle_index[-500:]:
            inter = len(sh & prev_sh)
            union = len(sh | prev_sh) or 1
            if inter / union >= 0.75:
                is_near_dup = True
                break
        if is_near_dup:
            counts["removed_near_duplicates_lsh"] += 1
            continue

        seen_prompts.add(p_norm)
        shingle_index.append(sh)

        canonical_sft = (
            f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{p_norm}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{th_norm}\n</think>\n{resp_norm}<|im_end|>"
        )
        doc_hash = hashlib.sha256(canonical_sft.encode("utf-8")).hexdigest()
        hasher.update(canonical_sft.encode("utf-8"))

        cleaned_rows.append({
            "doc_id": f"opus47-gold-{doc_hash[:16]}",
            "content_sha256": doc_hash,
            "source_dataset": row.get("source_dataset"),
            "source_idx": row.get("source_idx"),
            "system": "You are a helpful AI assistant.",
            "prompt": p_norm,
            "thinking": th_norm,
            "response": resp_norm,
            "text": canonical_sft,
            "model": "claude-opus-4-7"
        })

    counts["retained_gold_rows"] = len(cleaned_rows)
    gold_out = DATA_DIR / "reasoning_distill_claude_opus_4_7_max_session4_gold.jsonl"
    with open(gold_out, "w", encoding="utf-8") as f:
        for r in cleaned_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "shard_id": f"shard-opus47-gold-{hasher.hexdigest()[:12]}",
        "upstream_source": "https://huggingface.co/datasets/lordx64/reasoning-distill-claude-opus-4-7-max",
        "model_reference": "https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled",
        "license": "apache-2.0",
        "post_clean_content_sha256": hasher.hexdigest(),
        "statistics": counts
    }
    manifest_out = TARGET_DIR / "manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Cleaning pipeline finished! Summary:")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    run_full_pipeline()
