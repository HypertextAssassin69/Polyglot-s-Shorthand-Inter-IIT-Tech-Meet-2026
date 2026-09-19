import json
import torch
import numpy as np
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    
    best_model_path = "scratch/byt5_best_model.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load test instances
    byt5_test = []
    with open("data/processed/byt5_normalization_triplets.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == "test":
                byt5_test.append(d)
                
    # Sample 500
    np.random.seed(42)
    sample_indices = np.random.choice(len(byt5_test), min(500, len(byt5_test)), replace=False)
    sample_instances = [byt5_test[i] for i in sample_indices]
    
    t_tokenize = []
    t_generate = []
    t_decode = []
    t_e2e = []
    output_lengths = []
    
    print("Running 10 warmup iterations...")
    with torch.no_grad():
        for i in range(10):
            sample_text = "normalize: hello <n>wrld</n>"
            enc = tokenizer(sample_text, return_tensors="pt").to(device)
            out = model.generate(input_ids=enc.input_ids, max_new_tokens=20)
            _ = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    print("Benchmarking 500 iterations...")
    with torch.no_grad():
        for d in tqdm(sample_instances, desc="Benchmarking Latency"):
            input_text = d["input_text"]
            
            t_start = time.perf_counter()
            
            t_tok_0 = time.perf_counter()
            encoded = tokenizer(input_text, return_tensors="pt").to(device)
            t_tok_1 = time.perf_counter()
            
            gen_tokens = model.generate(input_ids=encoded.input_ids, max_new_tokens=20)
            t_gen_1 = time.perf_counter()
            
            pred = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            t_dec_1 = time.perf_counter()
            
            t_tokenize.append((t_tok_1 - t_tok_0) * 1000)
            t_generate.append((t_gen_1 - t_tok_1) * 1000)
            t_decode.append((t_dec_1 - t_gen_1) * 1000)
            t_e2e.append((t_dec_1 - t_start) * 1000)
            output_lengths.append(len(gen_tokens[0]))
            
    print("\n| Measurement | Time |")
    print(f"| --- | ---: |")
    print(f"| Tokenization | {np.mean(t_tokenize):.2f} ms |")
    print(f"| `generate()` | {np.mean(t_generate):.2f} ms |")
    print(f"| Decoding | {np.mean(t_decode):.2f} ms |")
    print(f"| **End-to-end** | **{np.mean(t_e2e):.2f} ms** |")
    print(f"| P50 | {np.percentile(t_e2e, 50):.2f} ms |")
    print(f"| P95 | {np.percentile(t_e2e, 95):.2f} ms |")
    print(f"\nAverage token length: {np.mean(output_lengths):.2f}")
    
    # Also print parameters
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()
