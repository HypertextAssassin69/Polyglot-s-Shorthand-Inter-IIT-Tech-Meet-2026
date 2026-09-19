import csv
import re
import sys
import os
import random
import time
from collections import defaultdict

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

def is_vowel(c): return c in 'aeiou'

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

def gen_f4_terminal(word):
    if not word: return set()
    res = {word}
    if word[-1:] in ['n', 'm', 'h']:
        res.add(word[:-1])
    else:
        res.add(word + 'n')
        res.add(word + 'm')
        res.add(word + 'h')
    return res

def gen_f5_repeated(word):
    if len(word) > 15: return {word}
    collapsed = re.sub(r'(.)\1+', r'\1', word)
    res = {word, collapsed}
    for i in range(len(word)):
        res.add(word[:i] + word[i] + word[i:])
    return res

def gen_f6_chat(word):
    mapping = {
        'plz': 'please', 'plzz': 'please', 'pls': 'please', 'u': 'you', 'ur': 'your', 
        'govt': 'government', 'gov': 'government', 'no': 'number', 'q': 'kyun', 'v': 'bhi',
        'bro': 'brother', 'msg': 'message', 'sis': 'sister', 'sry': 'sorry', 'thnx': 'thanks',
        'h': 'hai', 'k': 'ke', 'm': 'mein', 'ni': 'nahi', 'nai': 'nahi', 'kr': 'kar'
    }
    if word in mapping: return {word, mapping[word]}
    return {word}

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
    print("Loading data directly...", flush=True)
    vocab = set()
    missing_pairs = defaultdict(int)
    
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
                missing_pairs[(a,b)] += 1
            
    missing_pairs = {k: v for k, v in missing_pairs.items() if k[0] != k[1]}
    total_missing = sum(missing_pairs.values())
    
    print(f"Total Unique Pairs: {len(missing_pairs)}, Total missing instances: {total_missing}", flush=True)
    vocab_list = list(vocab)
    
    print("\n" + "="*50)
    print("EXPERIMENT A: F4/F5/F6 UNION")
    print("="*50)
    
    v_sample = random.sample(vocab_list, min(10000, len(vocab_list)))
    def col(f):
        u = defaultdict(list)
        for w in v_sample:
            for c in f(w): u[c].append(w)
        c = sum(1 for v in u.values() if len(v) > 1)
        return int(c * (len(vocab_list)/len(v_sample)))
        
    def f4(w): return gen_f4_terminal(w)
    def f45(w): 
        r = set()
        for c in gen_f4_terminal(w): r.update(gen_f5_repeated(c))
        return r
    def f456(w):
        r = set()
        for c in gen_f4_terminal(w):
            for c2 in gen_f5_repeated(c):
                r.update(gen_f6_chat(c2))
        return r
        
    cf4 = col(f4)
    cf45 = col(f45)
    cf456 = col(f456)
    
    r4, r45, r456 = 0, 0, 0
    c4_sizes, c45_sizes, c456_sizes = [], [], []
    
    for (n, g), count in missing_pairs.items():
        s4 = f4(n)
        s45 = f45(n)
        s456 = f456(n)
        if g in s4: r4 += count
        if g in s45: r45 += count
        if g in s456: r456 += count
        c4_sizes.append(len(s4))
        c45_sizes.append(len(s45))
        c456_sizes.append(len(s456))
        
    print(f"1. F4: {r4} ({r4/total_missing:.2%}) | Marg: {r4/total_missing:.2%} | Col: {cf4} | Cands: {sum(c4_sizes)/len(c4_sizes):.1f} med {sorted(c4_sizes)[len(c4_sizes)//2]} max {max(c4_sizes)}", flush=True)
    print(f"2. F4+F5: {r45} ({r45/total_missing:.2%}) | Marg: {(r45-r4)/total_missing:.2%} | Col: {cf45} | Cands: {sum(c45_sizes)/len(c45_sizes):.1f} med {sorted(c45_sizes)[len(c45_sizes)//2]} max {max(c45_sizes)}", flush=True)
    print(f"3. F4+F5+F6: {r456} ({r456/total_missing:.2%}) | Marg: {(r456-r45)/total_missing:.2%} | Col: {cf456} | Cands: {sum(c456_sizes)/len(c456_sizes):.1f} med {sorted(c456_sizes)[len(c456_sizes)//2]} max {max(c456_sizes)}", flush=True)
    
    print("\n" + "="*50)
    print("EXPERIMENT B: WEIGHTED RETRIEVAL")
    print("="*50)
    
    sample_pairs = random.sample(list(missing_pairs.items()), min(100, len(missing_pairs)))
    
    def eval_retrieval(dist_func, name):
        t0 = time.time()
        r1, r3, r5, r10, r20 = 0, 0, 0, 0, 0
        total = sum(c for _, c in sample_pairs)
        fails = []
        for (n, g), count in sample_pairs:
            dists = [(w, dist_func(n, w)) for w in vocab_list]
            dists.sort(key=lambda x: x[1])
            rank = -1
            for i, (w, d) in enumerate(dists):
                if w == g:
                    rank = i + 1
                    break
            if rank != -1:
                if rank <= 1: r1 += count
                if rank <= 3: r3 += count
                if rank <= 5: r5 += count
                if rank <= 10: r10 += count
                if rank <= 20: r20 += count
            if rank > 20 or rank == -1:
                fails.append((n, g, count, dists[:5]))
                
        t1 = time.time()
        print(f"\nMethod: {name}", flush=True)
        print(f"R@1: {r1/total:.2%} | R@3: {r3/total:.2%} | R@5: {r5/total:.2%} | R@10: {r10/total:.2%} | R@20: {r20/total:.2%}", flush=True)
        print(f"Total time: {t1-t0:.2f}s ({((t1-t0)/len(sample_pairs)):.4f}s per query)", flush=True)
        return fails

    fails_std = eval_retrieval(std_dist, "Baseline (Standard Levenshtein)")
    fails_w = eval_retrieval(w_dist, "Weighted Transliteration Distance")
    
    print("\nFailure Analysis (Weighted Top 20 miss):")
    fails_w.sort(key=lambda x: x[2], reverse=True)
    for n, g, c, top in fails_w[:15]:
        top_str = ", ".join(f"{w}({d:.1f})" for w, d in top)
        print(f"Miss: {n} -> {g} ({c}). Top retrieved: {top_str}")

if __name__ == "__main__":
    main()
