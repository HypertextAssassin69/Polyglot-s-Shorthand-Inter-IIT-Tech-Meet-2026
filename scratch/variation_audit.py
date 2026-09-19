import pandas as pd
import re
import sys
import os
import random
import itertools
from collections import defaultdict
import numpy as np

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import build_phonetic_index, generate_candidates

RAW_DATA_PATH = "data/raw/comi_lingua/TN_train.csv"

def get_changed_word_pairs(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    if len(w1) != len(w2): return []
    pairs = []
    for a, b in zip(w1, w2):
        a_clean = re.sub(r'([!?.]){2,}', ' ', a)
        b_clean = re.sub(r'([!?.]){2,}', ' ', b)
        a_clean = re.sub(r'[^\w\s]', '', a_clean).lower()
        b_clean = re.sub(r'[^\w\s]', '', b_clean).lower()
        if a_clean and b_clean and a_clean != b_clean:
            pairs.append((a_clean, b_clean))
    return pairs

def is_vowel(c):
    return c in 'aeiou'

def gen_f1_vowel_insert_delete(word):
    if len(word) > 10: return set() # Prevent combinatorial explosion
    chars = list(word)
    paths = [[]]
    for i, c in enumerate(chars):
        next_paths = []
        for p in paths:
            next_paths.append(p + [c])
            if not is_vowel(c) and p and not is_vowel(p[-1]):
                next_paths.append(p + ['a', c])
            if c == 'a':
                next_paths.append(p)
        paths = next_paths[:5000] # Cap paths to prevent OOM
    final_paths = []
    for p in paths:
        final_paths.append("".join(p))
        if p and not is_vowel(p[-1]):
            final_paths.append("".join(p + ['a']))
    return set(f for f in final_paths if f != word and f != "")

def gen_f2_vowel_length(word):
    if len(word) > 15: return set()
    subs = {'a': ['aa'], 'aa': ['a'], 'e': ['ai', 'a'], 'ai': ['e'], 
            'i': ['ee'], 'ee': ['i'], 'u': ['oo'], 'oo': ['u'], 'o': ['au'], 'au': ['o']}
    paths = [[]]
    i = 0
    while i < len(word):
        next_paths = []
        if i + 1 < len(word):
            c2 = word[i:i+2]
            if c2 in subs:
                for p in paths:
                    next_paths.append(p + [c2])
                    for s in subs[c2]: next_paths.append(p + [s])
                paths = next_paths[:5000]
                i += 2
                continue
        c1 = word[i]
        for p in paths:
            next_paths.append(p + [c1])
            if c1 in subs:
                for s in subs[c1]: next_paths.append(p + [s])
        paths = next_paths[:5000]
        i += 1
    return set("".join(p) for p in paths if "".join(p) != word and "".join(p) != "")

def gen_f3_consonant(word):
    if len(word) > 15: return set()
    subs = {'v': ['w', 'bh'], 'w': ['v'], 'z': ['j'], 'j': ['z'], 's': ['sh'], 'sh': ['s'], 
            'f': ['ph'], 'ph': ['f'], 'c': ['k'], 'x': ['ksh']}
    paths = [[]]
    i = 0
    while i < len(word):
        next_paths = []
        if i + 1 < len(word):
            c2 = word[i:i+2]
            if len(c2) == 2 and c2[1] == 'h' and c2[0] not in 'aeiou':
                for p in paths:
                    next_paths.append(p + [c2])
                    next_paths.append(p + [c2[0]])
                paths = next_paths[:5000]
                i += 2
                continue
            if c2 in subs:
                for p in paths:
                    next_paths.append(p + [c2])
                    for s in subs[c2]: next_paths.append(p + [s])
                paths = next_paths[:5000]
                i += 2
                continue
        c1 = word[i]
        for p in paths:
            next_paths.append(p + [c1])
            if c1 in subs:
                for s in subs[c1]: next_paths.append(p + [s])
            if not is_vowel(c1) and c1 != 'h':
                next_paths.append(p + [c1 + 'h'])
        paths = next_paths[:5000]
        i += 1
    return set("".join(p) for p in paths if "".join(p) != word and "".join(p) != "")

def gen_f4_terminal(word):
    if not word: return set()
    res = set()
    if word[-1:] in ['n', 'm', 'h']:
        res.add(word[:-1])
    else:
        res.add(word + 'n')
        res.add(word + 'm')
        res.add(word + 'h')
    return {r for r in res if r != word and r != ""}

def gen_f5_repeated(word):
    if len(word) > 15: return set()
    collapsed = re.sub(r'(.)\1+', r'\1', word)
    res = {collapsed}
    for i in range(len(word)):
        res.add(word[:i] + word[i] + word[i:])
    return {r for r in res if r != word and r != ""}

def gen_f6_chat(word):
    mapping = {
        'plz': 'please', 'plzz': 'please', 'pls': 'please', 'u': 'you', 'ur': 'your', 
        'govt': 'government', 'gov': 'government', 'no': 'number', 'q': 'kyun', 'v': 'bhi',
        'bro': 'brother', 'msg': 'message', 'sis': 'sister', 'sry': 'sorry', 'thnx': 'thanks',
        'h': 'hai', 'k': 'ke', 'm': 'mein', 'ni': 'nahi', 'nai': 'nahi', 'kr': 'kar'
    }
    if word in mapping: return {mapping[word]}
    return set()

FAMILIES = [
    ("F1. Vowel/Schwa", gen_f1_vowel_insert_delete),
    ("F2. Vowel Length", gen_f2_vowel_length),
    ("F3. Consonant", gen_f3_consonant),
    ("F4. Terminal N/H", gen_f4_terminal),
    ("F5. Repeated Ltr", gen_f5_repeated),
    ("F6. Chat Dict", gen_f6_chat)
]

def run_variation_audit():
    print("Loading data...", flush=True)
    df = pd.read_csv(RAW_DATA_PATH)
    
    pair_counts = defaultdict(int)
    vocab = set()
    for _, row in df.iterrows():
        try:
            ns = str(row['Sentences'])
            gs = str(row['Annotated by: Annotator 1'])
            for w in ns.split():
                cw = re.sub(r'[^\w\s]', '', w).lower()
                if cw: vocab.add(cw)
        except:
            continue
        pairs = get_changed_word_pairs(ns, gs)
        for p in pairs:
            pair_counts[p] += 1
            
    print("Finding missing pairs...", flush=True)
    index = build_phonetic_index()
    missing_pairs = {}
    for (n, g), count in pair_counts.items():
        cands = set(generate_candidates(n, index))
        if g not in cands:
            missing_pairs[(n, g)] = count
            
    total_missing = sum(missing_pairs.values())
    print(f"Total Missing Pairs: {total_missing}", flush=True)
    
    results = []
    
    vocab_sample = random.sample(list(vocab), min(5000, len(vocab))) # sample 5k to speed up
    
    for fname, func in FAMILIES:
        print(f"\n--- Testing {fname} ---", flush=True)
        
        recovered = 0
        recovered_examples = []
        for (n, g), count in missing_pairs.items():
            try:
                cands = func(n)
                if g in cands:
                    recovered += count
                    recovered_examples.append((n, g, count))
            except: pass
                
        pct = (recovered / 57188) * 100
        
        cand_sizes = []
        unique_targets = defaultdict(list)
        
        for w in vocab_sample:
            try:
                cands = func(w)
                cand_sizes.append(len(cands))
                for c in cands:
                    unique_targets[c].append(w)
            except:
                cand_sizes.append(0)
                
        collisions = sum(1 for c, sources in unique_targets.items() if len(sources) > 1)
        # Normalize collisions to 38k vocab estimate
        collisions_est = int(collisions * (len(vocab) / len(vocab_sample)))
        
        avg_g = np.mean(cand_sizes)
        med_g = np.median(cand_sizes)
        p95_g = np.percentile(cand_sizes, 95)
        max_g = np.max(cand_sizes)
        
        recovered_examples.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Recovered: {recovered} / 57188 ({pct:.2f}%)", flush=True)
        print(f"Candidate Growth: Avg {avg_g:.2f}, Med {med_g}, P95 {p95_g}, Max {max_g}", flush=True)
        print(f"Collisions (Est on full vocab): {collisions_est}", flush=True)
        print(f"Top Recovered Examples:")
        for n, g, c in recovered_examples[:10]:
            print(f"  {n} -> {g} ({c})")
            
        major_failure = "Candidate explosion" if avg_g > 10 else "Low recall" if pct < 5 else "Collisions" if collisions_est > 500 else "None"
        if fname == "F6. Chat Dict": major_failure = "Maintenance overhead"
        if fname == "F1. Vowel/Schwa" and avg_g > 10: major_failure = "Explosion/Collisions"
        
        results.append({
            'Family': fname,
            'Recovered': recovered,
            'Pct': pct,
            'Collisions': collisions_est,
            'AvgGrowth': avg_g,
            'FailureMode': major_failure
        })

    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Family':<20} | {'Recoverable':<12} | {'%':<6} | {'Collisions':<10} | {'Avg Growth':<10} | {'Major Failure Mode':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['Family']:<20} | {r['Recovered']:<12} | {r['Pct']:<6.2f} | {r['Collisions']:<10} | {r['AvgGrowth']:<10.2f} | {r['FailureMode']:<20}")

if __name__ == "__main__":
    run_variation_audit()
