import pandas as pd
import re
import sys
import os
import random

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import phonetic_tokenize

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

def is_vowel_phoneme(p):
    return p in ['A', 'AA', 'I', 'EE', 'U', 'OO', 'E', 'AI', 'O', 'AU', 'a', 'e', 'i', 'o', 'u']

def get_plausible_phoneme_paths(word):
    base = phonetic_tokenize(word)
    paths = set()
    current_paths = [[]]
    for i, p in enumerate(base):
        next_paths = []
        for cp in current_paths:
            if cp and not is_vowel_phoneme(cp[-1]) and not is_vowel_phoneme(p):
                next_paths.append(cp + ['a', p])
                next_paths.append(cp + ['AA', p]) 
            if p == 'e': next_paths.append(cp + ['AI'])
            if p == 'u': next_paths.append(cp + ['OO'])
            if p == 'i': next_paths.append(cp + ['EE'])
            if p == 'a': next_paths.append(cp + ['AA'])
            if p == 'v': next_paths.append(cp + ['BH'])
            next_paths.append(cp + [p])
        current_paths = next_paths
        
    for cp in current_paths:
        paths.add(tuple(cp))
        if cp and cp[-1] in ['n', 'h', 'm']:
            paths.add(tuple(cp[:-1]))
    return paths

def run_extraction():
    df = pd.read_csv(RAW_DATA_PATH)
    vocab = set()
    pair_counts = {}
    
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
            pair_counts[p] = pair_counts.get(p, 0) + 1

    # Part A: Successful and Failed examples
    successes = []
    failures = []
    
    sampled_pairs = random.sample(list(pair_counts.items()), min(2000, len(pair_counts)))
    for (noisy, gold), count in sampled_pairs:
        gold_ph = tuple(phonetic_tokenize(gold))
        paths = get_plausible_phoneme_paths(noisy)
        if gold_ph in paths:
            successes.append((noisy, gold, gold_ph, count))
        else:
            failures.append((noisy, gold, gold_ph, list(paths)[:2], count))
            
    successes.sort(key=lambda x: x[3], reverse=True)
    failures.sort(key=lambda x: x[4], reverse=True)
    
    print("--- SUCCESSFUL EXAMPLES ---")
    for s in successes[:15]:
        print(f"Noisy: {s[0]:<10} Gold: {s[1]:<10} | Ph: {s[2]}")
        
    print("\n--- FAILED EXAMPLES ---")
    for f in failures[:15]:
        print(f"Noisy: {f[0]:<10} Gold: {f[1]:<10} | Gold Ph: {f[2]} | Sample Path: {f[3][0]}")

    # Part C: Heuristic examples breakdown
    english_abbr = {'plz', 'govt', 'u', 'ur', 'thnx', 'sry', 'msg', 'bro', 'sis', 'hw', 'no', 'ok', 'okkk'}
    chat_words = [w for w in vocab if w in english_abbr or re.search(r'\d', w) or len(re.sub(r'[aeiou]', '', w)) == len(w)]
    
    print(f"\n--- CHAT FORMS BREAKDOWN (Total: {len(chat_words)}) ---")
    sample_chat = random.sample(chat_words, min(50, len(chat_words)))
    for w in sample_chat:
        print(w)

if __name__ == "__main__":
    run_extraction()
