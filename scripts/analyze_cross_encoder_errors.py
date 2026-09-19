import json
import random

HYBRID_PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/hybrid_predictions.jsonl"
MLP_PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/mlp_predictions.jsonl"
CE_PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/cross_encoder_predictions.jsonl"

def load_preds(path):
    preds = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds

def main():
    h_preds = load_preds(HYBRID_PRED_PATH)
    m_preds = load_preds(MLP_PRED_PATH)
    c_preds = load_preds(CE_PRED_PATH)
    
    assert len(h_preds) == len(c_preds), "Prediction counts do not match!"
    
    dot_fails_ce_succeeds = []
    dot_succeeds_ce_fails = []
    both_fail = []
    
    for i in range(len(h_preds)):
        h = h_preds[i]
        c = c_preds[i]
        
        # Only care about instances where retriever succeeded (gold is present)
        if h['target_idx'] == -1:
            continue
            
        h_correct = h['is_correct']
        c_correct = c['is_correct']
        
        if not h_correct and c_correct:
            dot_fails_ce_succeeds.append((h, c))
        elif h_correct and not c_correct:
            dot_succeeds_ce_fails.append((h, c))
        elif not h_correct and not c_correct:
            both_fail.append((h, c))
            
    print(f"Total Ranker cases (Gold present): {len(h_preds) - sum(1 for h in h_preds if h['target_idx'] == -1)}")
    print(f"Hybrid V1 (Dot) Ranker Fails: {sum(1 for h in h_preds if h['target_idx'] != -1 and not h['is_correct'])}")
    print(f"Cross-Encoder Ranker Fails: {sum(1 for c in c_preds if c['target_idx'] != -1 and not c['is_correct'])}")
    
    print(f"\nDot fails, CE succeeds (Fixed): {len(dot_fails_ce_succeeds)}")
    print(f"Dot succeeds, CE fails (Broken): {len(dot_succeeds_ce_fails)}")
    print(f"Both fail: {len(both_fail)}\n")
    
    random.seed(42)
    sample = random.sample(dot_fails_ce_succeeds, min(30, len(dot_fails_ce_succeeds)))
    
    print("--- 30 Random Examples Fixed by Cross-Encoder ---")
    for i, (h, c) in enumerate(sample):
        print(f"[{i+1}] Sentence: {h['raw_sentence']}")
        print(f"Noisy: {h['noisy_word']}  |  Gold: {h['target_word']}")
        print(f"Cands: {h['candidates']}")
        print(f"Hybrid V1 (Dot Product) Pred: {h['predicted_word']} (Wrong)")
        print(f"Cross-Encoder Pred: {c['predicted_word']} (FIXED!)")
        
        # Formatted scores
        gold_idx = h['target_idx']
        pred_idx_h = h['predicted_idx']
        
        print(f"  CE Score (Gold {h['target_word']}): {c['all_scores'][gold_idx]:.4f}")
        print(f"  CE Score (Distractor {h['predicted_word']}): {c['all_scores'][pred_idx_h]:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
