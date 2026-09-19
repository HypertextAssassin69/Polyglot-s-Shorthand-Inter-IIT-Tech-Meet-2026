import json
import torch
import numpy as np
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    
    best_model_path = "scratch/byt5_best_model.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from {best_model_path}")
    else:
        raise Exception(f"Checkpoint not found at {best_model_path}!")
        
    model.to(device)
    model.eval()

    # 1. Load ByT5 test instances
    byt5_test = []
    with open("data/processed/byt5_normalization_triplets.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == "test":
                byt5_test.append(d)
                
    # 2. Load Hybrid V1 candidates to determine Covered/Missed
    hybrid_v1_cands = {}
    with open("data/processed/v1_hybrid_triplets.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['raw_sentence']}_{d['word_index']}_{d['noisy_word']}"
            hybrid_v1_cands[key] = d["candidates"]
            
    print(f"Loaded {len(byt5_test)} ByT5 test instances.")
    
    overall_correct = 0
    noisy_correct = 0
    noisy_total = 0
    clean_correct = 0
    clean_total = 0
    covered_correct = 0
    covered_total = 0
    missed_correct = 0
    missed_total = 0
    
    t_tokenize = []
    t_generate = []
    t_decode = []
    t_e2e = []
    output_lengths = []
    
    errors = []
    
    # WARMUP
    print("Running 10 warmup iterations...")
    with torch.no_grad():
        for i in range(10):
            sample_text = "normalize: hello <n>wrld</n>"
            enc = tokenizer(sample_text, return_tensors="pt").to(device)
            out = model.generate(input_ids=enc.input_ids, max_new_tokens=20)
            _ = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    print("Evaluating...")
    with torch.no_grad():
        for idx, d in enumerate(tqdm(byt5_test, desc="Testing")):
            input_text = d["input_text"]
            target_text = d["target_text"]
            is_noisy = d["is_noisy"]
            key = f"{d['raw_sentence']}_{d['word_index']}_{d['noisy_word']}"
            
            # Start E2E timer
            t_start = time.perf_counter()
            
            # 1. Tokenize
            t_tok_0 = time.perf_counter()
            encoded = tokenizer(input_text, return_tensors="pt").to(device)
            t_tok_1 = time.perf_counter()
            
            # 2. Generate
            gen_tokens = model.generate(input_ids=encoded.input_ids, max_new_tokens=20)
            t_gen_1 = time.perf_counter()
            
            # 3. Decode
            pred = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            t_dec_1 = time.perf_counter()
            
            # Measure
            t_tokenize.append((t_tok_1 - t_tok_0) * 1000)
            t_generate.append((t_gen_1 - t_tok_1) * 1000)
            t_decode.append((t_dec_1 - t_gen_1) * 1000)
            t_e2e.append((t_dec_1 - t_start) * 1000)
            output_lengths.append(len(gen_tokens[0]))
            
            is_correct = (pred == target_text.strip())
            
            if is_correct:
                overall_correct += 1
            else:
                errors.append({
                    "raw_sentence": d["raw_sentence"],
                    "noisy_word": d["noisy_word"],
                    "target_word": target_text,
                    "prediction": pred,
                    "is_noisy": is_noisy,
                    "input_text": input_text,
                    "hybrid_v1_cands": hybrid_v1_cands.get(key, [])
                })
                
            if is_noisy:
                noisy_total += 1
                if is_correct: noisy_correct += 1
                
                # Check coverage in Hybrid V1
                if key in hybrid_v1_cands:
                    cands = hybrid_v1_cands[key]
                    if target_text in cands:
                        covered_total += 1
                        if is_correct: covered_correct += 1
                    else:
                        missed_total += 1
                        if is_correct: missed_correct += 1
            else:
                clean_total += 1
                if is_correct: clean_correct += 1

    # Save errors for analysis
    with open("scratch/byt5_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print("\n=== EXPERIMENT F EVALUATION RESULTS ===")
    print(f"Overall Exact Match Accuracy: {overall_correct / len(byt5_test):.2%} (N={len(byt5_test)})")
    
    if noisy_total > 0:
        print(f"Noisy-word Correction Accuracy (input ≠ gold): {noisy_correct / noisy_total:.2%} (N={noisy_total})")
    
    if clean_total > 0:
        print(f"Clean-word Preservation Accuracy (input = gold): {clean_correct / clean_total:.2%} (N={clean_total})")
        
    if covered_total > 0:
        print(f"Retrieval-Covered Accuracy (gold in V1 candidate set): {covered_correct / covered_total:.2%} (N={covered_total})")
        
    if missed_total > 0:
        print(f"Retrieval-Missed Accuracy (gold NOT in V1 candidate set): {missed_correct / missed_total:.2%} (N={missed_total})")
        
    print("\n=== INFERENCE LATENCY ===")
    print(f"| Measurement | Time |")
    print(f"| --- | ---: |")
    print(f"| Tokenization | {np.mean(t_tokenize):.2f} ms |")
    print(f"| `generate()` | {np.mean(t_generate):.2f} ms |")
    print(f"| Decoding | {np.mean(t_decode):.2f} ms |")
    print(f"| **End-to-end** | **{np.mean(t_e2e):.2f} ms** |")
    print(f"| P50 | {np.percentile(t_e2e, 50):.2f} ms |")
    print(f"| P95 | {np.percentile(t_e2e, 95):.2f} ms |")
    
    print(f"\nAverage generated output token length: {np.mean(output_lengths):.1f} tokens")
    
if __name__ == "__main__":
    main()
