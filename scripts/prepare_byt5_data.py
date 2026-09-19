import pandas as pd
import json
import os
import sys
import random
from difflib import SequenceMatcher

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import clean_word

DATA_PATH = "data/raw/comi_lingua/TN_train.csv"
OUTPUT_PATH = "data/processed/byt5_normalization_triplets.jsonl"

def is_lexical_substitution(crw, cnw):
    if crw == cnw:
        return False
    
    # Exclude pairs that are completely different words (lexical substitution/paraphrase)
    # Using a simple SequenceMatcher ratio.
    # 'nhi' -> 'na' ratio is 0.4. 'kya' -> 'baaton' ratio is 0.22.
    # 'ordr' -> 'order' ratio is 0.88.
    ratio = SequenceMatcher(None, crw, cnw).ratio()
    return ratio < 0.5

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
    
    splits = {}
    for s in unique_sentences[:train_end]: splits[s] = 'train'
    for s in unique_sentences[train_end:val_end]: splits[s] = 'val'
    for s in unique_sentences[val_end:]: splits[s] = 'test'
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    total_noisy = 0
    total_clean = 0
    total_excluded = 0
    
    # First pass: count global ratio in the whole corpus
    print("Measuring natural clean : noisy ratio...")
    global_clean_count = 0
    global_noisy_count = 0
    
    valid_noisy_instances = []
    valid_clean_instances = []
    
    for ns in all_sentences:
        gs = sentence_pairs[ns]
        raw_words = ns.split()
        norm_words = gs.split()
        
        for i, (rw, nw) in enumerate(zip(raw_words, norm_words)):
            crw = clean_word(rw)
            cnw = clean_word(nw)
            if not crw or not cnw: continue
            
            if crw != cnw:
                if is_lexical_substitution(crw, cnw):
                    total_excluded += 1
                    continue
                global_noisy_count += 1
                
                context_words = list(raw_words)
                context_words[i] = f"<n>{crw}</n>"
                input_text = "normalize: " + " ".join(context_words)
                
                valid_noisy_instances.append({
                    "raw_sentence": ns,
                    "word_index": i,
                    "noisy_word": crw,
                    "target_word": cnw,
                    "is_noisy": True,
                    "input_text": input_text,
                    "target_text": cnw,
                    "split": splits[ns]
                })
            else:
                global_clean_count += 1
                context_words = list(raw_words)
                context_words[i] = f"<n>{crw}</n>"
                input_text = "normalize: " + " ".join(context_words)
                
                valid_clean_instances.append({
                    "raw_sentence": ns,
                    "word_index": i,
                    "noisy_word": crw,
                    "target_word": cnw,
                    "is_noisy": False,
                    "input_text": input_text,
                    "target_text": cnw,
                    "split": splits[ns]
                })

    natural_ratio = global_clean_count / max(1, global_noisy_count)
    print(f"Global Noisy Words: {global_noisy_count}")
    print(f"Global Clean Words: {global_clean_count}")
    print(f"Natural Clean:Noisy Ratio: {natural_ratio:.2f}:1")
    print(f"Excluded Lexical Substitutions: {total_excluded}")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for sample in valid_noisy_instances:
            out_f.write(json.dumps(sample) + "\n")
            total_noisy += 1
            
        for sample in valid_clean_instances:
            out_f.write(json.dumps(sample) + "\n")
            total_clean += 1
            
    print(f"Total correction (noisy) instances saved: {total_noisy}")
    print(f"Total preservation (clean) instances saved: {total_clean}")
    print(f"Total instances: {total_noisy + total_clean}")
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dataset()
