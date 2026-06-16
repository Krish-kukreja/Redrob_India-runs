"""
This testing file is here to look and test how dense string extraction is working and how to improve it 
"""

import json
import sys
import os

# Makes sure that i can see files in main path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractors import CandidateExtractor

def test_dense_string():
    extractor = CandidateExtractor(reference_date_str="2026-06-15")

    with open("data/sample_candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Grab the first candidate
    first_candidate = candidates[32]
    candidate_id = first_candidate.get("candidate_id")
    
    print(f"\n--- TESTING CANDIDATE: {candidate_id} ---\n")

    # --- DEBUGGING THE SUMMARY (Fluff Shredding) ---
    raw_summary = first_candidate.get('profile', {}).get('summary', '')
    summary_debug = extractor._extract_high_signal_sentences(raw_summary, is_summary=True)
    
    print("=== SUMMARY BUCKETS ===")
    print(f"TRASH (Fluff):   {summary_debug['trash']}")
    print(f"IMPACT (Kept):   {summary_debug['impact']}")
    print(f"ROUTINE (Kept):  {summary_debug['routine']}\n")

    # --- DEBUGGING THE CURRENT ROLE (Positional Priming) ---
    raw_desc = first_candidate.get('career_history', [{}])[0].get('description', '')
    desc_debug = extractor._extract_high_signal_sentences(raw_desc, is_summary=False)
    
    print("=== JOB DESCRIPTION BUCKETS ===")
    print(f"IMPACT (Pulled to Front): {desc_debug['impact']}")
    print(f"ROUTINE (Pushed to Back): {desc_debug['routine']}\n")

    # --- FINAL DENSE STRING ---
    dense_string = extractor.build_dense_string(first_candidate)
    
    print("=== FINAL OPTIMIZED DENSE STRING ===")
    print(dense_string)
    print("\n====================================\n")
    
    word_count = len(dense_string.split())
    print(f"Word Count: {word_count} words (Safe for Bi-Encoder: {word_count < 350})")

def test_rich_string():
    """Tests the comprehensive extraction for the Cross-Encoder."""
    extractor = CandidateExtractor(reference_date_str="2026-06-15")

    with open("data/sample_candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Grab the first candidate
    first_candidate = candidates[0]
    candidate_id = first_candidate.get("candidate_id", "UNKNOWN")
    
    print(f"\n--- TESTING RICH STRING: {candidate_id} ---\n")

    # --- VISIBILITY: DEBUGGING THE BUCKETS ---
    career = first_candidate.get('career_history', [])
    if career:
        print("=== JOB DESCRIPTION BUCKET TRIAGE (RICH STRING) ===")
        raw_desc = career[0].get('description', '')
        desc_debug = extractor._extract_high_signal_sentences(raw_desc, is_summary=False)
        
        print("🚀 IMPACT BUCKET (Metrics Pulled to Front):")
        if desc_debug['impact']:
            for idx, sent in enumerate(desc_debug['impact']):
                print(f"   {idx+1}. {sent}")
        else:
            print("   [No impact metrics detected]")
            
        print("\n📄 ROUTINE BUCKET (Trailing Context):")
        if desc_debug['routine']:
            for idx, sent in enumerate(desc_debug['routine']):
                print(f"   {idx+1}. {sent}")
        else:
            print("   [No routine sentences detected]")
        print("================================================\n")

    # --- FINAL RICH STRING ---
    rich_string = extractor.build_rich_string(first_candidate)
    
    print("=== FINAL RICH STRING (CROSS-ENCODER FORMAT) ===")
    print(rich_string)
    print("\n================================================\n")
    
    word_count = len(rich_string.split())
    print(f"Word Count: {word_count} words (Safe for Bi-Encoder: {word_count < 280})")

def run_full_extraction_test():
    print("\nInitializing Search Engine Extractor...")
    # Using a static reference date so days_inactive math is perfectly consistent during testing
    extractor = CandidateExtractor(reference_date_str="2026-06-15")

    try:
        with open("data/sample_candidates.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print("ERROR: data/sample_candidates.json not found. Make sure the path is correct.")
        return

    # Grab the first candidate to test the pipeline
    candidate = candidates[0]
    candidate_id = candidate.get("candidate_id", "UNKNOWN")

    print(f"\n================================================")
    print(f" PIPELINE DIAGNOSTICS: {candidate_id}")
    print(f"================================================\n")

    # --- 1. THE DENSE STRING ---
    print("--- 1. DENSE STRING (Bi-Encoder / Top 500 Recall) ---")
    dense_string = extractor.build_dense_string(candidate)
    print(dense_string)
    print(f"\n[Metrics] Characters: {len(dense_string)} | Approx Tokens: {len(dense_string.split()) * 1.3:.0f}")
    print("\n" + "-"*50 + "\n")

    # --- 2. THE RICH STRING ---
    print("--- 2. RICH STRING (Cross-Encoder / Top 50 Precision) ---")
    rich_string = extractor.build_rich_string(candidate)
    print(rich_string)
    print(f"\n[Metrics] Characters: {len(rich_string)} | Approx Tokens: {len(rich_string.split()) * 1.3:.0f}")
    print("\n" + "-"*50 + "\n")

    # --- 3. THE BEHAVIORAL TELEMETRY ---
    print("--- 3. TELEMETRY MULTIPLIERS (The Reality Check) ---")
    behavioral_data = extractor.extract_behavioral_signals(candidate)
    
    # Isolate just the 'telemetry' block and pretty-print it for the terminal
    clean_telemetry = behavioral_data.get("telemetry", {})
    print(json.dumps(clean_telemetry, indent=4))
    print("\n================================================\n")

if __name__ == "__main__":
    run_full_extraction_test()