import csv
import re
import sys
import os
import random
from collections import defaultdict

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
            if c1 == c2: sub_cost = 0.0
            elif (c1, c2) in subs_low: sub_cost = 0.2
            else: sub_cost = 1.0
                
            del_cost = 0.2 if c1 == 'a' or (i > 1 and c1 == w1[i-2]) else 1.0
            ins_cost = 0.2 if c2 == 'a' or (j > 1 and c2 == w2[j-2]) else 1.0
            dp[i][j] = min(dp[i-1][j] + del_cost, dp[i][j-1] + ins_cost, dp[i-1][j-1] + sub_cost)
    return dp[m][n]

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

def main():
    vocab = set()
    missing_pairs = []
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ns = str(row['Sentences'])
            gs = str(row['Annotated by: Annotator 1'])
            for w in ns.split():
                cw = re.sub(r'[^\w\s]', '', w).lower()
                if cw: vocab.add(cw)
            for w in gs.split():
                cw = re.sub(r'[^\w\s]', '', w).lower()
                if cw: vocab.add(cw)
            pairs = get_changed_word_pairs(ns, gs)
            for a, b in pairs:
                if a != b:
                    missing_pairs.append((a, b))
                    
    vocab_list = list(vocab)
    sample_pairs = random.sample(missing_pairs, min(20, len(missing_pairs)))
    
    print("Top-5 Distance Ties Check:", flush=True)
    for n, g in sample_pairs:
        dists = [(w, w_dist(n, w)) for w in vocab_list]
        dists.sort(key=lambda x: x[1])
        top_dists = [d for w, d in dists[:10]]
        print(f"Query: {n} -> Gold: {g} | Top 5 dists: {top_dists[:5]}")
        # check ties for rank 1
        rank1_dist = dists[0][1]
        ties_for_rank1 = sum(1 for w, d in dists if d == rank1_dist)
        print(f"  Ties for Rank 1 distance ({rank1_dist}): {ties_for_rank1} candidates")

if __name__ == "__main__":
    main()
