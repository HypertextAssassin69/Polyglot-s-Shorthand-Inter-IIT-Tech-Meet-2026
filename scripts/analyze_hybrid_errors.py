import json
import random

PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/hybrid_predictions.jsonl"

def main():
    errors = []
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            if not p["is_correct"]:
                errors.append(p)
                
    print(f"Total Errors found in predictions: {len(errors)}")
    
    random.seed(42)
    sample = random.sample(errors, min(30, len(errors)))
    
    for i, p in enumerate(sample):
        print(f"\n[{i+1}] Error Type: {p['error_type']}")
        print(f"Sentence: {p['raw_sentence']}")
        print(f"Noisy: {p['noisy_word']}")
        print(f"Gold : {p['target_word']}")
        print(f"Pred : {p['predicted_word']}")
        print(f"Cands: {p['candidates']}")

if __name__ == "__main__":
    main()
