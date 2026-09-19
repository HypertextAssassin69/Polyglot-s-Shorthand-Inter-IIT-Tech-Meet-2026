import pandas as pd
import re
import sys
import os
import random
from collections import defaultdict

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import build_phonetic_index, generate_candidates

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

def is_vowel(c):
    return c in 'aeiou'

def classify(n, g):
    n_len = len(n)
    g_len = len(g)
    
    # 9. English / code-mixed
    english_abbr = {'plz', 'plzz', 'plzzz', 'govt', 'u', 'ur', 'thnx', 'sry', 'msg', 'bro', 'sis', 'hw', 'no'}
    english_words = {'please', 'government', 'you', 'your', 'sorry', 'message', 'number'}
    if n in english_abbr or g in english_words or n == 'thanku':
        return "9. English/code-mixed abbreviation"
        
    # 10. Phonetic lexical / chat extreme
    lexical = {'q': 'kyun', 'v': 'bhi', 'ni': 'nahi', 'nai': 'nahi', 'k': 'ke', 'h': 'hai', 'm': 'mein'}
    if n in lexical and g == lexical[n]:
        return "10. phonetic lexical substitution"
        
    # 6. Nasalization / terminal n/h
    if n + 'n' == g or n + 'm' == g or n + 'h' == g or g + 'n' == n or g + 'h' == n:
        return "6. nasalization / terminal n/h variation"
        
    # 7. Repeated letter
    # reduce repeats
    n_red = re.sub(r'(.)\1+', r'\1', n)
    g_red = re.sub(r'(.)\1+', r'\1', g)
    if n_red == g_red and n != g:
        return "7. repeated-letter variation"
        
    # 4. Vowel deletion / schwa deletion / chat shorthand like rha
    # If consonants match exactly but vowels are missing in n
    n_cons = [c for c in n if not is_vowel(c)]
    g_cons = [c for c in g if not is_vowel(c)]
    if n_cons == g_cons and sum(1 for c in n if is_vowel(c)) < sum(1 for c in g if is_vowel(c)):
        if n in {'nhi', 'rha', 'rhe', 'rhi', 'hm', 'bhut', 'bhot', 'kya', 'kyu'}:
            return "8. abbreviation / chat shorthand"
        return "4. vowel deletion / schwa deletion"
        
    # 3. Aspiration variation
    # 'h' added or removed next to a consonant
    n_no_h = n.replace('h', '')
    g_no_h = g.replace('h', '')
    if n_no_h == g_no_h and n_no_h != n and n_no_h != g:
        return "3. aspiration variation"
    
    if n_no_h == g or n == g_no_h:
        return "3. aspiration variation"
        
    # 1. Vowel-length / vowel spelling variation
    if n_cons == g_cons:
        return "1. vowel-length / vowel spelling variation"
        
    # 2. Consonant spelling variation
    n_vowels = [c for c in n if is_vowel(c)]
    g_vowels = [c for c in g if is_vowel(c)]
    if n_vowels == g_vowels and len(n_cons) == len(g_cons):
        return "2. consonant spelling variation"
        
    # 11. Multi-character phonetic
    # e.g. x -> ksh, c -> k
    if 'x' in n or 'c' in n or 'sh' in n or 'ch' in n:
        # weak heuristic
        return "11. multi-character phonetic substitution"
        
    # 5. Consonant deletion or insertion
    if len(n_cons) != len(g_cons):
        if n in {'kr', 'pr', 'bta'}:
            return "8. abbreviation / chat shorthand"
        return "5. consonant deletion or insertion"

    return "13. other / unclear"

def run_taxonomy():
    df = pd.read_csv(RAW_DATA_PATH)
    index = build_phonetic_index()
    
    missing = defaultdict(int)
    
    pair_counts = defaultdict(int)
    for i, row in df.iterrows():
        try:
            ns = str(row['Sentences'])
            gs = str(row['Annotated by: Annotator 1'])
        except:
            continue
        pairs = get_changed_word_pairs(ns, gs)
        for p in pairs:
            pair_counts[p] += 1
            
    total_missing = 0
    for (n, g), count in pair_counts.items():
        cands = set(generate_candidates(n, index))
        if g not in cands:
            missing[(n, g)] = count
            total_missing += count
            
    cats = defaultdict(list)
    cat_counts = defaultdict(int)
    
    for (n, g), count in missing.items():
        cat = classify(n, g)
        cats[cat].append((n, g, count))
        cat_counts[cat] += count
        
    print(f"Total Missing Examples: {total_missing}")
    
    for cat in sorted(cats.keys()):
        count = cat_counts[cat]
        pct = (count / total_missing) * 100
        print(f"\n{cat.upper()}")
        print(f"Count: {count} ({pct:.2f}%)")
        
        # Sort by frequency for representative examples
        exs = sorted(cats[cat], key=lambda x: x[2], reverse=True)[:20]
        for n, g, c in exs:
            print(f"  {n:<15} -> {g:<15} ({c})")

if __name__ == "__main__":
    run_taxonomy()
