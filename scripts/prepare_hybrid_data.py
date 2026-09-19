import pandas as pd
import json
import os
import sys
import re
import random
from collections import defaultdict
import numpy as np
from numba import njit
import time

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import representation, RULES, clean_word

DATA_PATH = "data/raw/comi_lingua/TN_train.csv"
OUTPUT_PATH = "data/processed/v1_hybrid_triplets.jsonl"

@njit
def fast_w_dist(w1_arr, w2_arr):
    m = len(w1_arr)
    n = len(w2_arr)
    dp = np.zeros((m + 1, n + 1), dtype=np.float32)
    
    for i in range(1, m + 1):
        c1 = w1_arr[i-1]
        del_cost = 0.2 if c1 == 97 or (i > 1 and c1 == w1_arr[i-2]) else 1.0
        dp[i, 0] = dp[i-1, 0] + del_cost
        
    for j in range(1, n + 1):
        c2 = w2_arr[j-1]
        ins_cost = 0.2 if c2 == 97 or (j > 1 and c2 == w2_arr[j-2]) else 1.0
        dp[0, j] = dp[0, j-1] + ins_cost
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1 = w1_arr[i-1]
            c2 = w2_arr[j-1]
            
            if c1 == c2:
                sub_cost = 0.0
            elif (c1 == 105 and c2 == 101) or (c1 == 101 and c2 == 105) or \
                 (c1 == 117 and c2 == 111) or (c1 == 111 and c2 == 117) or \
                 (c1 == 101 and c2 == 97)  or (c1 == 97  and c2 == 101) or \
                 (c1 == 105 and c2 == 121) or (c1 == 121 and c2 == 105):
                sub_cost = 0.2
            else:
                sub_cost = 1.0
                
            del_cost = 0.2 if c1 == 97 or (i > 1 and c1 == w1_arr[i-2]) else 1.0
            ins_cost = 0.2 if c2 == 97 or (j > 1 and c2 == w2_arr[j-2]) else 1.0
            
            dp[i, j] = min(dp[i-1, j] + del_cost, dp[i, j-1] + ins_cost, dp[i-1, j-1] + sub_cost)
            
    return dp[m, n]

def encode_word(w):
    return np.array([ord(c) for c in w], dtype=np.int32)

def gen_f6_chat(word):
    mapping = {
        'plz': 'please', 'plzz': 'please', 'pls': 'please', 'u': 'you', 'ur': 'your', 
        'govt': 'government', 'gov': 'government', 'no': 'number', 'q': 'kyun', 'v': 'bhi',
        'bro': 'brother', 'msg': 'message', 'sis': 'sister', 'sry': 'sorry', 'thnx': 'thanks',
        'h': 'hai', 'k': 'ke', 'm': 'mein', 'ni': 'nahi', 'nai': 'nahi', 'kr': 'kar'
    }
    if word in mapping: return {mapping[word]}
    return set()

def prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    all_sentences = []
    sentence_pairs = {}
    
    for raw_sentence, normalized_sentence in zip(df["Sentences"], df["Annotated by: Annotator 1"]):
        if pd.isna(raw_sentence) or pd.isna(normalized_sentence): continue
        ns = str(raw_sentence)
        gs = str(normalized_sentence)
        if len(ns.split()) == len(gs.split()):
            all_sentences.append(ns)
            sentence_pairs[ns] = gs

    unique_sentences = list(set(all_sentences))
    unique_sentences.sort()
    random.seed(42)
    random.shuffle(unique_sentences)
    
    n_sents = len(unique_sentences)
    train_end = int(0.8 * n_sents)
    val_end = int(0.9 * n_sents)
    
    train_sents = set(unique_sentences[:train_end])
    val_sents = set(unique_sentences[train_end:val_end])
    test_sents = set(unique_sentences[val_end:])
    
    print(f"Train/Val/Test sentences: {len(train_sents)} / {len(val_sents)} / {len(test_sents)}")
    
    train_vocab = set()
    for ns in train_sents:
        gs = sentence_pairs[ns]
        for w in ns.split():
            cw = clean_word(w)
            if cw: train_vocab.add(cw)
        for w in gs.split():
            cw = clean_word(w)
            if cw: train_vocab.add(cw)
            
    print(f"Train Vocab Size (Retrieval Scope): {len(train_vocab)}")
    
    phonetic_index = defaultdict(set)
    for w in train_vocab:
        phonetic_index[representation(w)].add(w)
        
    vocab_list = list(train_vocab)
    encoded_vocab = [encode_word(w) for w in vocab_list]
    
    def get_phonetic_candidates(word):
        tokens = representation(word)
        cands = set()
        cands.update(phonetic_index.get(tokens, set()))
        for a, b in RULES:
            c1 = tuple(b if t == a else t for t in tokens)
            cands.update(phonetic_index.get(c1, set()))
            c2 = tuple(a if t == b else t for t in tokens)
            cands.update(phonetic_index.get(c2, set()))
        return cands

    # Precalculate Top 5 Weighted Dist for all unique noisy words across all splits
    # to avoid redundant calculations.
    unique_noisy_words = set()
    for ns in all_sentences:
        gs = sentence_pairs[ns]
        for nw, gw in zip(ns.split(), gs.split()):
            cnw = clean_word(nw)
            cgw = clean_word(gw)
            if cnw and cgw and cnw != cgw:
                unique_noisy_words.add(cnw)
                
    print(f"Precalculating Top-5 retrieval for {len(unique_noisy_words)} unique noisy words...")
    retrieval_cache = {}
    
    t0 = time.time()
    for idx, nw in enumerate(list(unique_noisy_words)):
        enw = encode_word(nw)
        # Vectorized dist calc using list comprehension calling the numba function
        dists = [fast_w_dist(enw, ev) for ev in encoded_vocab]
        
        # Combine and sort, filtering out self
        scored = [(vocab_list[i], d) for i, d in enumerate(dists) if vocab_list[i] != nw]
        scored.sort(key=lambda x: x[1])
        retrieval_cache[nw] = set(w for w, d in scored[:5])
        
        if (idx+1) % 1000 == 0:
            print(f"  Processed {idx+1}/{len(unique_noisy_words)} (Time: {time.time()-t0:.1f}s)")
            
    print(f"Retrieval precalculation complete in {time.time()-t0:.1f}s.")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    cand_counts = []
    
    r_A = 0
    r_B = 0
    r_C = 0
    r_AB = 0
    r_ABC = 0
    total_misses = 0
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for ns in all_sentences:
            gs = sentence_pairs[ns]
            raw_words = ns.split()
            norm_words = gs.split()
            
            for i, (rw, nw) in enumerate(zip(raw_words, norm_words)):
                crw = clean_word(rw)
                cnw = clean_word(nw)
                
                if not crw or not cnw or crw == cnw:
                    continue
                    
                total_misses += 1
                
                cand_A = get_phonetic_candidates(crw)
                cand_B = retrieval_cache[crw]
                cand_C = gen_f6_chat(crw)
                
                if cnw in cand_A: r_A += 1
                if cnw in cand_B: r_B += 1
                if cnw in cand_C: r_C += 1
                if cnw in cand_A or cnw in cand_B: r_AB += 1
                if cnw in cand_A or cnw in cand_B or cnw in cand_C: r_ABC += 1
                
                union_cands = list(cand_A | cand_B | cand_C)
                cand_counts.append(len(union_cands))
                
                sample = {
                    "raw_sentence": ns,
                    "word_index": i,
                    "noisy_word": crw,
                    "target_word": cnw,
                    "candidates": union_cands,
                    "cand_A_len": len(cand_A),
                    "cand_B_len": len(cand_B),
                    "cand_C_len": len(cand_C)
                }
                out_f.write(json.dumps(sample) + "\n")
                
    cand_counts.sort()
    mean_c = sum(cand_counts) / len(cand_counts)
    med_c = cand_counts[len(cand_counts)//2]
    p95_c = cand_counts[int(len(cand_counts)*0.95)]
    max_c = cand_counts[-1]
    
    print("\n=== HYBRID CANDIDATE STATISTICS ===")
    print(f"Mean Cands: {mean_c:.2f}")
    print(f"Median Cands: {med_c}")
    print(f"P95 Cands: {p95_c}")
    print(f"Max Cands: {max_c}")
    
    for k in range(1, 7):
        if k == 6:
            c = sum(1 for x in cand_counts if x >= 6)
            print(f"6+ cands: {c/len(cand_counts):.2%}")
        else:
            c = sum(1 for x in cand_counts if x == k)
            print(f"{k} cand(s): {c/len(cand_counts):.2%}")
            
    print(f"\nGold Present (Hybrid): {r_ABC/total_misses:.2%}")
    print(f"Gold Absent (Hybrid): {1 - r_ABC/total_misses:.2%}")
    
    print("\n=== CANDIDATE RECALL ABLATION ===")
    print(f"Total instances requiring normalization: {total_misses}")
    print(f"A (Strict Phonetic): {r_A/total_misses:.2%}")
    print(f"B (Weighted Retrieval): {r_B/total_misses:.2%}")
    print(f"C (Chat Dict): {r_C/total_misses:.2%}")
    print(f"D (Strict + Weighted): {r_AB/total_misses:.2%}")
    print(f"E (Strict + Weighted + Chat): {r_ABC/total_misses:.2%}")
    print(f"\nData saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dataset()
