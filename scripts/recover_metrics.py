import json

def recover_metrics():
    # 1. Load ByT5 test instances to get totals
    total = 0
    noisy_total = 0
    clean_total = 0
    covered_total = 0
    missed_total = 0
    
    hybrid_v1_cands = {}
    with open("data/processed/v1_hybrid_triplets.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['raw_sentence']}_{d['word_index']}_{d['noisy_word']}"
            hybrid_v1_cands[key] = d["candidates"]
            
    with open("data/processed/byt5_normalization_triplets.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == "test":
                total += 1
                is_noisy = d["is_noisy"]
                if is_noisy:
                    noisy_total += 1
                    key = f"{d['raw_sentence']}_{d['word_index']}_{d['noisy_word']}"
                    if key in hybrid_v1_cands:
                        cands = hybrid_v1_cands[key]
                        if d["target_text"] in cands:
                            covered_total += 1
                        else:
                            missed_total += 1
                else:
                    clean_total += 1
                    
    # 2. Count errors
    overall_err = 0
    noisy_err = 0
    clean_err = 0
    covered_err = 0
    missed_err = 0
    
    with open("scratch/byt5_errors.json", "r", encoding="utf-8") as f:
        errors = json.load(f)
        overall_err = len(errors)
        for e in errors:
            if e["is_noisy"]:
                noisy_err += 1
                target = e["target_word"]
                cands = e["hybrid_v1_cands"]
                if target in cands:
                    covered_err += 1
                else:
                    missed_err += 1
            else:
                clean_err += 1
                
    print("\n=== RECOVERED METRICS ===")
    print(f"Overall Exact Match Accuracy: {(total - overall_err) / total:.2%} (N={total})")
    print(f"Noisy-word Correction Accuracy: {(noisy_total - noisy_err) / noisy_total:.2%} (N={noisy_total})")
    print(f"Clean-word Preservation Accuracy: {(clean_total - clean_err) / clean_total:.2%} (N={clean_total})")
    print(f"Retrieval-Covered Accuracy: {(covered_total - covered_err) / covered_total:.2%} (N={covered_total})")
    print(f"Retrieval-Missed Accuracy: {(missed_total - missed_err) / missed_total:.2%} (N={missed_total})")

if __name__ == "__main__":
    recover_metrics()
