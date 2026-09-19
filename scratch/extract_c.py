import pandas as pd
import re
import sys
import os
import random

RAW_DATA_PATH = "data/raw/comi_lingua/TN_train.csv"

def run_extraction_c():
    df = pd.read_csv(RAW_DATA_PATH)
    vocab = set()
    
    for _, row in df.iterrows():
        try:
            ns = str(row['Sentences'])
            for w in ns.split():
                cw = re.sub(r'[^\w\s]', '', w).lower()
                if cw: vocab.add(cw)
        except:
            continue

    # Part C: Heuristic examples breakdown
    english_abbr = {'plz', 'govt', 'u', 'ur', 'thnx', 'sry', 'msg', 'bro', 'sis', 'hw', 'no', 'ok', 'okkk'}
    
    chat_words = []
    for w in vocab:
        # Must be purely alphabetical ascii to be chat shorthand / english
        if not re.match(r'^[a-z]+$', w):
            continue
        # Is it in abbreviation list?
        if w in english_abbr:
            chat_words.append((w, "2. English/code-mixed abbreviation"))
            continue
        # Does it lack vowels entirely? (Consonant-only shorthand)
        if not re.search(r'[aeiou]', w):
            chat_words.append((w, "1. chat shorthand (vowel-less)"))
            continue

    # To get other categories (vowel/schwa deletion, phonetic spelling, lexical substitution), 
    # we need the noisy->gold mapping pairs where they mismatch.
    pair_counts = {}
    for _, row in df.iterrows():
        try:
            ns = str(row['Sentences'])
            gs = str(row['Annotated by: Annotator 1'])
            w1 = ns.split()
            w2 = gs.split()
            if len(w1) == len(w2):
                for a, b in zip(w1, w2):
                    a_clean = re.sub(r'[^\w\s]', '', a).lower()
                    b_clean = re.sub(r'[^\w\s]', '', b).lower()
                    if a_clean and b_clean and a_clean != b_clean and re.match(r'^[a-z]+$', a_clean) and re.match(r'^[a-z]+$', b_clean):
                        p = (a_clean, b_clean)
                        pair_counts[p] = pair_counts.get(p, 0) + 1
        except:
            pass

    # Sort pair counts by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    
    # We will pick 10 examples each for:
    # 3. vowel/schwa deletion
    # 4. phonetic spelling variation
    # 5. lexical substitution
    # 6. other normalization
    
    cat3 = []
    cat4 = []
    cat5 = []
    cat6 = []
    
    def is_vowel(c): return c in 'aeiou'
    lexical = {'q': 'kyun', 'v': 'bhi', 'ni': 'nahi', 'k': 'ke', 'h': 'hai', 'm': 'mein'}
    
    for (n, g), count in sorted_pairs:
        n_cons = [c for c in n if not is_vowel(c)]
        g_cons = [c for c in g if not is_vowel(c)]
        
        # Lexical
        if n in lexical and g == lexical[n]:
            cat5.append((n, g, count))
            continue
            
        # Vowel deletion
        if n_cons == g_cons and sum(1 for c in n if is_vowel(c)) < sum(1 for c in g if is_vowel(c)):
            cat3.append((n, g, count))
            continue
            
        # Phonetic spelling variation (same consonants, different vowels)
        if n_cons == g_cons:
            cat4.append((n, g, count))
            continue
            
        # Other (e.g. n/h variations, consonant variations)
        cat6.append((n, g, count))
            
    print("--- 1 & 2. CHAT / ENGLISH ABBREV ---")
    random.seed(42)
    for w, cat in random.sample(chat_words, min(20, len(chat_words))):
        print(f"{cat}: {w}")
        
    print("\n--- 3. VOWEL/SCHWA DELETION ---")
    for n, g, c in cat3[:10]:
        print(f"{n} -> {g} ({c})")
        
    print("\n--- 4. PHONETIC SPELLING VARIATION ---")
    for n, g, c in cat4[:10]:
        print(f"{n} -> {g} ({c})")
        
    print("\n--- 5. LEXICAL SUBSTITUTION ---")
    for n, g, c in cat5[:10]:
        print(f"{n} -> {g} ({c})")
        
    print("\n--- 6. OTHER NORMALIZATION ---")
    for n, g, c in cat6[:10]:
        print(f"{n} -> {g} ({c})")


if __name__ == "__main__":
    run_extraction_c()
