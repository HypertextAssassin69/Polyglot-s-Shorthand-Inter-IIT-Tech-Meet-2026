import csv
import re
import sys
import os
import random
import time
from collections import defaultdict
import numpy as np

# Fast edit distance fallback
def fast_std_dist(w1, w2):
    m, n = len(w1), len(w2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if w1[i-1] == w2[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = temp
    return dp[n]

try:
    import Levenshtein
    def std_dist(w1, w2): return Levenshtein.distance(w1, w2)
except:
    std_dist = fast_std_dist

RAW_DATA_PATH = "data/raw/comi_lingua/TN_train.csv"

def get_changed_word_pairs(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    if len(w1) != len(w2): return []
    pairs = []
    for a, b in zip(w1, w2):
        a_clean = re.sub(r'([!?.]){2,}', ' ', a).lower()
        b_clean = re.sub(r'([!?.]){2,}', ' ', b).lower()
        a_clean = re.sub(r'[^\w\s]', '', a_clean)
        b_clean = re.sub(r'[^\w\s]', '', b_clean)
        if a_clean and b_clean and a_clean != b_clean:
            pairs.append((a_clean, b_clean))
    return pairs

def w_dist(w1, w2):
    m, n = len(w1), len(w2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        c1 = w1[i-1]
        del_cost = 0.2 if c1 == 'a' or (i > 1 and c1 == w1[i-2]) else 1.0
        dp[i][0] = dp[i-1][0] + del_cost
    for j in range(1, n + 1):
        c2 = w2[j-1]
        ins_cost = 0.2 if c2 == 'a' or (j > 1 and c2 == w2[j-2]) else 1.0
        dp[0][j] = dp[0][j-1] + ins_cost
        
    subs_low = {('i','e'), ('e','i'), ('u','o'), ('o','u'), ('e','a'), ('a','e'), ('i','y'), ('y','i')} 
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1 = w1[i-1]
            c2 = w2[j-1]
            if c1 == c2:
                sub_cost = 0.0
            elif (c1, c2) in subs_low:
                sub_cost = 0.2
            else:
                sub_cost = 1.0
                
            del_cost = 0.2 if c1 == 'a' or (i > 1 and c1 == w1[i-2]) else 1.0
            ins_cost = 0.2 if c2 == 'a' or (j > 1 and c2 == w2[j-2]) else 1.0
            
            dp[i][j] = min(dp[i-1][j] + del_cost, dp[i][j-1] + ins_cost, dp[i-1][j-1] + sub_cost)
    return dp[m][n]

def main():
    print("Loading data and creating splits...", flush=True)
    all_sentences = []
    sentence_pairs = {}
    
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ns = str(row['Sentences'])
            gs = str(row['Annotated by: Annotator 1'])
            all_sentences.append(ns)
            sentence_pairs[ns] = gs
            
    unique_sentences = list(set(all_sentences))
    unique_sentences.sort()
    random.seed(42)
    random.shuffle(unique_sentences)
    
    n = len(unique_sentences)
    train_sents = unique_sentences[:int(0.8*n)]
    test_sents = unique_sentences[int(0.9*n):]
    
    # 1. Build Train Vocabulary
    train_vocab = set()
    for ns in train_sents:
        gs = sentence_pairs[ns]
        for w in ns.split():
            cw = re.sub(r'[^\w\s]', '', w).lower()
            if cw: train_vocab.add(cw)
        for w in gs.split():
            cw = re.sub(r'[^\w\s]', '', w).lower()
            if cw: train_vocab.add(cw)
            
    # 2. Build Test Queries
    test_queries = defaultdict(int)
    for ns in test_sents:
        gs = sentence_pairs[ns]
        pairs = get_changed_word_pairs(ns, gs)
        for a, b in pairs:
            test_queries[(a, b)] += 1
            
    total_test_misses = sum(test_queries.values())
    
    # Check OOV
    oov_count = 0
    in_vocab_queries = {}
    for (n_w, g_w), count in test_queries.items():
        if g_w not in train_vocab:
            oov_count += count
        else:
            in_vocab_queries[(n_w, g_w)] = count
            
    print(f"Train Vocab Size: {len(train_vocab)}")
    print(f"Total Test Queries: {total_test_misses}")
    print(f"Test Targets OOV: {oov_count} ({oov_count/total_test_misses:.2%})")
    print(f"Test Targets In-Vocab: {total_test_misses - oov_count} ({(total_test_misses - oov_count)/total_test_misses:.2%})\n")
    
    vocab_list = list(train_vocab)
    # Exclude queries where noisy word is the gold word (just in case)
    sample_size = min(300, len(test_queries))
    sample_pairs = random.sample(list(test_queries.items()), sample_size)
    
    def eval_retrieval(dist_func, name):
        t0 = time.time()
        r1, r3, r5, r10, r20 = 0, 0, 0, 0, 0
        in_v_r1, in_v_r3, in_v_r5, in_v_r10, in_v_r20 = 0, 0, 0, 0, 0
        total = sum(c for _, c in sample_pairs)
        in_vocab_total = sum(c for (n, g), c in sample_pairs if g in train_vocab)
        
        fails = []
        for (n_w, g_w), count in sample_pairs:
            # calculate distance to all vocab
            dists = [(w, dist_func(n_w, w)) for w in vocab_list if w != n_w] # EXCLUDE SELF
            dists.sort(key=lambda x: x[1])
            rank = -1
            for i, (w, d) in enumerate(dists):
                if w == g_w:
                    rank = i + 1
                    break
            if rank != -1:
                if rank <= 1: r1 += count; in_v_r1 += count
                if rank <= 3: r3 += count; in_v_r3 += count
                if rank <= 5: r5 += count; in_v_r5 += count
                if rank <= 10: r10 += count; in_v_r10 += count
                if rank <= 20: r20 += count; in_v_r20 += count
            if rank > 5 or rank == -1:
                fails.append((n_w, g_w, count, dists[:5]))
                
        t1 = time.time()
        print(f"\n--- {name} ---")
        print("OVERALL RECALL:")
        print(f"R@1: {r1/total:.2%} | R@3: {r3/total:.2%} | R@5: {r5/total:.2%} | R@10: {r10/total:.2%} | R@20: {r20/total:.2%}")
        if in_vocab_total > 0:
            print("IN-VOCAB RECALL:")
            print(f"R@1: {in_v_r1/in_vocab_total:.2%} | R@3: {in_v_r3/in_vocab_total:.2%} | R@5: {in_v_r5/in_vocab_total:.2%} | R@10: {in_v_r10/in_vocab_total:.2%} | R@20: {in_v_r20/in_vocab_total:.2%}")
        print(f"Time: {t1-t0:.2f}s ({((t1-t0)/len(sample_pairs)):.4f}s per query)")
        return fails

    print("Evaluating Baseline...")
    fails_std = eval_retrieval(std_dist, "Baseline (Standard Levenshtein)")
    print("Evaluating Weighted...")
    fails_w = eval_retrieval(w_dist, "Weighted Transliteration Distance")
    
    print("\nFailure Analysis (Weighted Top 5 miss):")
    fails_w.sort(key=lambda x: x[2], reverse=True)
    for n_w, g_w, c, top in fails_w[:30]:
        top_str = ", ".join(f"{w}({d:.1f})" for w, d in top)
        oov_flag = "[OOV]" if g_w not in train_vocab else ""
        print(f"Miss: {n_w} -> {g_w} {oov_flag}. Top retrieved: {top_str}")

if __name__ == "__main__":
    main()
