import pandas as pd
import re
import sys
import os
import random
from collections import defaultdict
import itertools

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
    """
    Approximation of a Roman->phoneme transducer lattice.
    Given a noisy word like 'krna', we generate paths by:
    1. Base tokenization
    2. Allowing optional schwa ('a' or 'A') insertion between adjacent consonants.
    3. Allowing terminal 'n' / 'h' drops.
    This simulates what a weighted lattice would do.
    """
    base = phonetic_tokenize(word)
    
    # Generate variations by inserting 'a' between adjacent consonants
    paths = set()
    
    # We will build paths incrementally
    current_paths = [[]]
    
    for i, p in enumerate(base):
        next_paths = []
        for cp in current_paths:
            # If last was consonant and current is consonant, optionally insert schwa
            if cp and not is_vowel_phoneme(cp[-1]) and not is_vowel_phoneme(p):
                next_paths.append(cp + ['a', p])
                next_paths.append(cp + ['AA', p]) # long vowel option
            
            # Allow common vowel swaps
            if p == 'e':
                next_paths.append(cp + ['AI'])
            if p == 'u':
                next_paths.append(cp + ['OO'])
            if p == 'i':
                next_paths.append(cp + ['EE'])
            if p == 'a':
                next_paths.append(cp + ['AA'])
                
            # Allow replacing chat 'v' with 'bh'
            if p == 'v':
                next_paths.append(cp + ['BH'])
                
            next_paths.append(cp + [p])
            
        current_paths = next_paths
        
    for cp in current_paths:
        paths.add(tuple(cp))
        # Optional terminal drops
        if cp and cp[-1] in ['n', 'h', 'm']:
            paths.add(tuple(cp[:-1]))
            
    return paths

def run_feasibility():
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Lexicon Coverage Stats
    vocab = set()
    for row in df['Sentences'].dropna():
        for w in row.split():
            cw = re.sub(r'[^\w\s]', '', w).lower()
            if cw:
                vocab.add(cw)
                
    total_vocab = len(vocab)
    print(f"Total Unique TN_train Vocabulary: {total_vocab}")
    
    # Heuristics for dictionary vs chat
    english_abbr = {'plz', 'govt', 'u', 'ur', 'thnx', 'sry', 'msg', 'bro', 'sis', 'hw', 'no', 'ok', 'okkk'}
    chat_forms = [w for w in vocab if w in english_abbr or re.search(r'\d', w) or len(re.sub(r'[aeiou]', '', w)) == len(w)]
    
    print(f"Approximate Chat/Code-Mixed/Reduced: {len(chat_forms)} ({len(chat_forms)/total_vocab:.2%})")
    print(f"Approximate Formal (remainder): {total_vocab - len(chat_forms)} ({(total_vocab - len(chat_forms))/total_vocab:.2%})")
    
    # Gold Phoneme-Path Recall
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
            
    total_pairs = sum(pair_counts.values())
    recall_count = 0
    failures = []
    
    # Sample 5000 pairs to save computation time
    sampled_pairs = random.sample(list(pair_counts.items()), min(5000, len(pair_counts)))
    sampled_total = sum(count for _, count in sampled_pairs)
    
    print(f"\nEvaluating Phoneme-Path Recall on sample of {sampled_total} pairs...")
    
    for (noisy, gold), count in sampled_pairs:
        # SELF-REFERENTIAL BOOTSTRAPPING: 
        # Using Roman->phoneme logic as a proxy for Canonical G2P of the gold word.
        canonical_gold_phonemes = tuple(phonetic_tokenize(gold))
        
        plausible_paths = get_plausible_phoneme_paths(noisy)
        
        if canonical_gold_phonemes in plausible_paths:
            recall_count += count
        else:
            failures.append((noisy, gold, count, canonical_gold_phonemes, list(plausible_paths)[:3]))
            
    print(f"Gold Phoneme-Path Recall: {recall_count} / {sampled_total} ({recall_count/sampled_total:.2%})")
    
    # Sort failures by frequency
    failures.sort(key=lambda x: x[2], reverse=True)
    print("\nRepresentative Failures:")
    for f in failures[:15]:
        print(f"  Noisy: {f[0]:<10} Gold: {f[1]:<10} (Count: {f[2]}) | Gold Ph: {f[3]}")
        print(f"    Sample paths: {f[4]}")

if __name__ == "__main__":
    run_feasibility()
