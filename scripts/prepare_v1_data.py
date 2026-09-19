import pandas as pd
import json
import os
import sys

# Ensure imports work from project root
sys.path.append(os.getcwd())

from tests.phonetic_benchmark import (
    clean_word, build_phonetic_index, generate_candidates
)

DATA_PATH = "data/raw/comi_lingua/TN_train.csv"
OUTPUT_PATH = "data/processed/v1_candidate_ranking_triplets.jsonl"

def prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    index = build_phonetic_index()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    valid_samples = 0
    total_changed = 0
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for raw_sentence, normalized_sentence in zip(
            df["Sentences"], df["Annotated by: Annotator 1"]
        ):
            if pd.isna(raw_sentence) or pd.isna(normalized_sentence):
                continue
                
            raw_words = raw_sentence.split()
            normalized_words = normalized_sentence.split()
            
            # Skip misaligned sentences for this strict V1 evaluation
            if len(raw_words) != len(normalized_words):
                continue
                
            for i, (raw_word, normalized_word) in enumerate(zip(raw_words, normalized_words)):
                clean_raw = clean_word(raw_word)
                clean_norm = clean_word(normalized_word)
                
                if not clean_raw or not clean_norm:
                    continue
                    
                # We only want words that required normalization
                if clean_raw == clean_norm:
                    continue
                    
                total_changed += 1
                
                # Generate candidates using the existing phonetic machinery
                candidates = generate_candidates(clean_raw, index)
                candidate_list = sorted(candidates)
                
                # V1 Filter: Target MUST be in the generated candidates
                if clean_norm in candidate_list:
                    # Save as a valid V1 triplet
                    sample = {
                        "raw_sentence": raw_sentence,
                        "word_index": i,
                        "noisy_word": clean_raw,
                        "target_word": clean_norm,
                        "candidates": candidate_list
                    }
                    out_f.write(json.dumps(sample) + "\n")
                    valid_samples += 1

    print(f"Total changed words processed: {total_changed}")
    print(f"Valid V1 triplets extracted: {valid_samples}")
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dataset()
