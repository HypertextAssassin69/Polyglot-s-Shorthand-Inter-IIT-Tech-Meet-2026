import pandas as pd
import re
import sys
import os
import random
from collections import defaultdict

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import build_phonetic_index, generate_candidates, edit_distance

RAW_DATA_PATH = "data/raw/comi_lingua/TN_train.csv"

def get_changed_word_pairs(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    
    if len(w1) != len(w2):
        return []
        
    pairs = []
    for a, b in zip(w1, w2):
        a_clean = re.sub(r'([!?.]){2,}', ' ', a)
        b_clean = re.sub(r'([!?.]){2,}', ' ', b)
        
        a_clean = re.sub(r'[^\w\s]', '', a_clean).lower()
        b_clean = re.sub(r'[^\w\s]', '', b_clean).lower()
        
        if a_clean and b_clean and a_clean != b_clean:
            pairs.append((a_clean, b_clean))
    return pairs

def categorize_error(noisy, gold):
    n_len = len(noisy)
    g_len = len(gold)
    
    if n_len > g_len and noisy.startswith(gold):
        return "deletion/insertion"
    if g_len > n_len and gold.startswith(noisy):
        return "deletion/insertion"
        
    ed = edit_distance(noisy, gold)
    
    if abs(n_len - g_len) == 1 and ('h' in noisy or 'h' in gold):
        return "aspiration"
        
    vowels = set('aeiou')
    n_vowels = [c for c in noisy if c in vowels]
    g_vowels = [c for c in gold if c in vowels]
    if len(n_vowels) != len(g_vowels) and set(n_vowels) == set(g_vowels):
        return "vowel-length variation"
        
    if ed > 3:
        return "non-phonetic normalization"
        
    return "phonetic spelling drift"

def analyze_raw_coverage():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("Building phonetic index...")
    index = build_phonetic_index()
    
    total_pairs = 0
    present_count = 0
    absent_count = 0
    
    ed_dist = defaultdict(lambda: {"total": 0, "present": 0})
    size_dist = defaultdict(lambda: {"total": 0, "present": 0})
    
    missing_examples = []
    pair_counts = defaultdict(int)
    
    for i, row in df.iterrows():
        try:
            noisy_sent = str(row['Sentences'])
            gold_sent = str(row['Annotated by: Annotator 1'])
        except Exception:
            continue
            
        pairs = get_changed_word_pairs(noisy_sent, gold_sent)
        for p in pairs:
            pair_counts[p] += 1

    print(f"Total unique word pairs to process: {len(pair_counts)}")
    
    processed_pairs = 0
    for (noisy, gold), count in pair_counts.items():
        processed_pairs += 1
        
        cands = list(generate_candidates(noisy, index))
        is_present = (gold in cands)
        
        c_size = len(cands)
        ed = edit_distance(noisy, gold)
        
        total_pairs += count
        if is_present:
            present_count += count
        else:
            absent_count += count
            missing_examples.append((noisy, gold, count, ed, cands))
            
        ed_dist[ed]["total"] += count
        size_dist[c_size]["total"] += count
        
        if is_present:
            ed_dist[ed]["present"] += count
            size_dist[c_size]["present"] += count
            
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"1. Total aligned changed word pairs: {total_pairs}")
    print(f"2. Gold-present count: {present_count}")
    print(f"3. Gold-absent count: {absent_count}")
    
    if total_pairs > 0:
        print(f"4. Candidate-generation recall: {(present_count/total_pairs)*100:.2f}%")
    
    print("\n5. Breakdown by ED:")
    for ed in sorted(ed_dist.keys())[:10]:
        t = ed_dist[ed]["total"]
        p = ed_dist[ed]["present"]
        if t > 0:
            print(f"   ED {ed}: {(p/t)*100:.2f}% ({p}/{t})")
        
    print("\n6. Breakdown by Size:")
    for size in sorted(size_dist.keys())[:10]:
        t = size_dist[size]["total"]
        p = size_dist[size]["present"]
        if t > 0:
            print(f"   Size {size}: {(p/t)*100:.2f}% ({p}/{t})")
        
    missing_examples.sort(key=lambda x: x[2], reverse=True)
    
    groups = defaultdict(list)
    for m in missing_examples:
        noisy, gold, count, ed, cands = m
        cat = categorize_error(noisy, gold)
        groups[cat].append(m)
        
    print("\n8. Breakdown of Missing Cases by Category (top representatives):")
    rep_count = 0
    
    for cat, examples in groups.items():
        print(f"\n--- {cat.upper()} ---")
        for m in examples[:15]: 
            if rep_count >= 100:
                break
            noisy, gold, count, ed, cands = m
            print(f"   Noisy: {noisy:<15} Gold: {gold:<15} (Count: {count})")
            rep_count += 1
        if rep_count >= 100:
            break

if __name__ == "__main__":
    analyze_raw_coverage()
