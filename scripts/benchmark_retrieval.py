import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.append(os.getcwd())
from tests.phonetic_benchmark import representation, RULES, clean_word
from scripts.prepare_hybrid_data import fast_w_dist, encode_word, gen_f6_chat

DATA_PATH = "data/raw/comi_lingua/TN_train.csv"

def get_bigrams(w):
    w = "#" + w + "#"
    return [w[i:i+2] for i in range(len(w)-1)]

class NgramIndex:
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.encoded_vocab = [encode_word(w) for w in vocab_list]
        self.vocab_size = len(vocab_list)
        
        # Build index
        t0 = time.time()
        self.index = defaultdict(list)
        for i, w in enumerate(vocab_list):
            for bg in set(get_bigrams(w)):
                self.index[bg].append(i)
                
        # Convert lists to numpy arrays for faster concatenation later
        for bg in self.index:
            self.index[bg] = np.array(self.index[bg], dtype=np.int32)
            
        print(f"[NgramIndex] Indexed {self.vocab_size} words in {time.time()-t0:.3f}s")
        
    def query(self, query_word, top_m=100, top_k=5):
        # Extract unique query bigrams
        q_bgs = set(get_bigrams(query_word))
        
        # Gather arrays
        arrays = [self.index[bg] for bg in q_bgs if bg in self.index]
        
        if not arrays:
            return set()
            
        # Fast overlapping count using numpy
        concatenated = np.concatenate(arrays)
        counts = np.bincount(concatenated, minlength=self.vocab_size)
        
        # Get indices of top M overlapping words
        # (argpartition is O(N) instead of O(N log N) sort)
        if len(counts) > top_m:
            top_m_indices = np.argpartition(counts, -top_m)[-top_m:]
        else:
            top_m_indices = np.arange(len(counts))
            
        # Now compute fast_w_dist only on these top M candidates
        enw = encode_word(query_word)
        scored = []
        for idx in top_m_indices:
            w = self.vocab_list[idx]
            if w == query_word:
                continue
            d = fast_w_dist(enw, self.encoded_vocab[idx])
            scored.append((w, d))
            
        scored.sort(key=lambda x: x[1])
        return set(w for w, d in scored[:top_k])

def main():
    print("Loading data and splitting 80/10/10...")
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
    
    train_sents = set(unique_sentences[:train_end])
    test_sents = set(unique_sentences[val_end:])
    
    train_vocab = set()
    for ns in train_sents:
        gs = sentence_pairs[ns]
        for w in ns.split():
            cw = clean_word(w)
            if cw: train_vocab.add(cw)
        for w in gs.split():
            cw = clean_word(w)
            if cw: train_vocab.add(cw)
            
    vocab_list = list(train_vocab)
    encoded_vocab = [encode_word(w) for w in vocab_list]
    
    print("Building Phonetic Index...")
    phonetic_index = defaultdict(set)
    for w in train_vocab:
        phonetic_index[representation(w)].add(w)
        
    def get_phonetic_candidates(word):
        tokens = representation(word)
        cands = set()
        cands.update(phonetic_index.get(tokens, set()))
        for a, b in RULES:
            c1 = tuple(b if t == a else t for t in tokens)
            cands.update(phonetic_index.get(c1, set()))
            c2 = tuple(a if t == b else t for t in tokens)
            cands.update(phonetic_index.get(c2, set()))
        return cands
        
    def full_weighted_query(query_word):
        enw = encode_word(query_word)
        dists = [fast_w_dist(enw, ev) for ev in encoded_vocab]
        scored = [(vocab_list[i], d) for i, d in enumerate(dists) if vocab_list[i] != query_word]
        scored.sort(key=lambda x: x[1])
        return set(w for w, d in scored[:5])

    print("Building Ngram Index...")
    ngram_index = NgramIndex(vocab_list)
    
    print("Extracting Test Queries...")
    test_queries = []
    for ns in test_sents:
        gs = sentence_pairs[ns]
        for nw, gw in zip(ns.split(), gs.split()):
            cnw = clean_word(nw)
            cgw = clean_word(gw)
            if cnw and cgw and cnw != cgw:
                test_queries.append((cnw, cgw))
                
    print(f"Total Test Queries (noisy->gold pairs): {len(test_queries)}")
    
    # Warm up numba
    fast_w_dist(np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))
    
    # We will sample 1000 unique queries to measure empirical latency
    unique_noisy_test = list(set([q[0] for q in test_queries]))
    random.shuffle(unique_noisy_test)
    eval_queries = unique_noisy_test[:1000]
    
    print(f"Benchmarking latency on {len(eval_queries)} unique queries...")
    
    results = {}
    
    def benchmark_func(name, func, qs):
        latencies = []
        cands_dict = {}
        t_start = time.time()
        for q in qs:
            t0 = time.perf_counter()
            cands = func(q)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            cands_dict[q] = cands
        total_time = time.time() - t_start
        print(f"[{name}] Completed in {total_time:.2f}s")
        return latencies, cands_dict

    # 1. Strict Phonetic
    lat_phonetic, cands_phonetic = benchmark_func("Strict Phonetic", get_phonetic_candidates, eval_queries)
    
    # 2. Chat Dict
    lat_chat, cands_chat = benchmark_func("Chat Dict", gen_f6_chat, eval_queries)
    
    # 3. Indexed Retrieval
    lat_indexed_100, cands_indexed_100 = benchmark_func("Indexed Retrieval (M=100)", lambda q: ngram_index.query(q, top_m=100, top_k=5), eval_queries)
    lat_indexed_500, cands_indexed_500 = benchmark_func("Indexed Retrieval (M=500)", lambda q: ngram_index.query(q, top_m=500, top_k=5), eval_queries)
    lat_indexed_1000, cands_indexed_1000 = benchmark_func("Indexed Retrieval (M=1000)", lambda q: ngram_index.query(q, top_m=1000, top_k=5), eval_queries)
    
    # 4. Full Weighted
    lat_full, cands_full = benchmark_func("Full O(N) Weighted", full_weighted_query, eval_queries)
    
    print("Evaluating recall on ALL test queries...")
    
    def evaluate_config(name, latencies, cands_dict_func):
        recall_1 = 0
        recall_3 = 0
        recall_5 = 0
        recall_10 = 0
        absent = 0
        cand_counts = []
        
        t0 = time.time()
        for cnw, cgw in test_queries:
            cands = cands_dict_func(cnw)
            
            enw = encode_word(cnw)
            scored = [(w, fast_w_dist(enw, encode_word(w))) for w in cands]
            scored.sort(key=lambda x: x[1])
            ranked_cands = [w for w, d in scored]
            
            if cgw in ranked_cands[:1]: recall_1 += 1
            if cgw in ranked_cands[:3]: recall_3 += 1
            if cgw in ranked_cands[:5]: recall_5 += 1
            if cgw in ranked_cands[:10]: recall_10 += 1
            if cgw not in ranked_cands: absent += 1
                
            cand_counts.append(len(ranked_cands))
            
        total = len(test_queries)
        cand_counts.sort()
        
        res = {
            "name": name,
            "R@1": recall_1 / total,
            "R@3": recall_3 / total,
            "R@5": recall_5 / total,
            "R@10": recall_10 / total,
            "Absent": absent / total,
            "Mean Cands": np.mean(cand_counts),
            "Median Cands": cand_counts[len(cand_counts)//2],
            "P95 Cands": cand_counts[int(len(cand_counts)*0.95)],
            "Max Cands": cand_counts[-1],
            "Mean Latency (ms)": np.mean(latencies),
            "P50 Latency (ms)": np.percentile(latencies, 50),
            "P95 Latency (ms)": np.percentile(latencies, 95),
            "Total Eval Time (s)": time.time() - t0
        }
        print(f"  {name}: R@5={res['R@5']:.2%}, Latency={res['Mean Latency (ms)']:.2f}ms")
        return res

    configs = []
    
    configs.append(evaluate_config("Strict Phonetic", lat_phonetic, get_phonetic_candidates))
    configs.append(evaluate_config("Full Weighted", lat_full, lambda q: cands_full.get(q, full_weighted_query(q))))
    configs.append(evaluate_config("Chat Dict", lat_chat, gen_f6_chat))
    
    hybrid_lats = [lat_phonetic[i] + lat_chat[i] + lat_full[i] for i in range(len(eval_queries))]
    def hybrid_cands(q):
        s1 = get_phonetic_candidates(q)
        s2 = gen_f6_chat(q)
        s3 = cands_full.get(q, full_weighted_query(q))
        return s1 | s2 | s3
    configs.append(evaluate_config("Hybrid Union (Baseline)", hybrid_lats, hybrid_cands))
    
    configs.append(evaluate_config("Indexed Retrieval (M=100)", lat_indexed_100, lambda q: cands_indexed_100.get(q, ngram_index.query(q, top_m=100, top_k=5))))
    configs.append(evaluate_config("Indexed Retrieval (M=500)", lat_indexed_500, lambda q: cands_indexed_500.get(q, ngram_index.query(q, top_m=500, top_k=5))))
    configs.append(evaluate_config("Indexed Retrieval (M=1000)", lat_indexed_1000, lambda q: cands_indexed_1000.get(q, ngram_index.query(q, top_m=1000, top_k=5))))
    
    hybrid_idx_lats_100 = [lat_phonetic[i] + lat_chat[i] + lat_indexed_100[i] for i in range(len(eval_queries))]
    def hybrid_idx_cands_100(q):
        s1 = get_phonetic_candidates(q)
        s2 = gen_f6_chat(q)
        s3 = cands_indexed_100.get(q, ngram_index.query(q, top_m=100, top_k=5))
        return s1 | s2 | s3
    configs.append(evaluate_config("Indexed + Hybrid Union (M=100)", hybrid_idx_lats_100, hybrid_idx_cands_100))
    
    hybrid_idx_lats_500 = [lat_phonetic[i] + lat_chat[i] + lat_indexed_500[i] for i in range(len(eval_queries))]
    def hybrid_idx_cands_500(q):
        s1 = get_phonetic_candidates(q)
        s2 = gen_f6_chat(q)
        s3 = cands_indexed_500.get(q, ngram_index.query(q, top_m=500, top_k=5))
        return s1 | s2 | s3
    configs.append(evaluate_config("Indexed + Hybrid Union (M=500)", hybrid_idx_lats_500, hybrid_idx_cands_500))
    
    hybrid_idx_lats_1000 = [lat_phonetic[i] + lat_chat[i] + lat_indexed_1000[i] for i in range(len(eval_queries))]
    def hybrid_idx_cands_1000(q):
        s1 = get_phonetic_candidates(q)
        s2 = gen_f6_chat(q)
        s3 = cands_indexed_1000.get(q, ngram_index.query(q, top_m=1000, top_k=5))
        return s1 | s2 | s3
    configs.append(evaluate_config("Indexed + Hybrid Union (M=1000)", hybrid_idx_lats_1000, hybrid_idx_cands_1000))
    
    with open("scratch/retrieval_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=4)
        
    print("\nBenchmarking complete! Results saved to scratch/retrieval_benchmark.json")

if __name__ == "__main__":
    main()
