import json
import random

HYBRID_PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/hybrid_predictions.jsonl"
MLP_PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/mlp_predictions.jsonl"

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
    
    assert len(h_preds) == len(m_preds), "Prediction counts do not match!"
    
    h_ranker_fails = []
    fixed_by_mlp = 0
    new_mlp_fails = 0
    
    for i in range(len(h_preds)):
        h = h_preds[i]
        m = m_preds[i]
        
        if h['error_type'] == 'Ranker Failure':
            h_ranker_fails.append((h, m))
            if m['is_correct']:
                fixed_by_mlp += 1
                
        if h['is_correct'] and m['error_type'] == 'Ranker Failure':
            new_mlp_fails += 1
            
    print(f"Total Ranker Failures in Hybrid V1: {len(h_ranker_fails)}")
    print(f"Fixed by MLP: {fixed_by_mlp}")
    print(f"New Ranker Failures in MLP: {new_mlp_fails}")
    print(f"Net change in Ranker Failures: {new_mlp_fails - fixed_by_mlp}\n")
    
    random.seed(42)
    sample = random.sample(h_ranker_fails, min(30, len(h_ranker_fails)))
    
    for i, (h, m) in enumerate(sample):
        print(f"[{i+1}] Sentence: {h['raw_sentence']}")
        print(f"Noisy: {h['noisy_word']}  |  Gold: {h['target_word']}")
        print(f"Cands: {h['candidates']}")
        print(f"Hybrid V1 (Dot Product) Pred: {h['predicted_word']} (Wrong)")
        
        # Because we added dot_score back into mlp output, we can show both for m:
        gold_dot = m.get('dot_score_gold', 'N/A')
        pred_dot = m.get('dot_score_pred', 'N/A')
        
        gold_mlp = m.get('mlp_score_gold', 'N/A')
        pred_mlp = m.get('mlp_score_pred', 'N/A')
        
        # Format scores nicely
        def fs(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        
        if m['is_correct']:
            print(f"MLP Pred: {m['predicted_word']} (FIXED!)")
        else:
            print(f"MLP Pred: {m['predicted_word']} (Still Wrong)")
            
        print(f"  Gold candidate -> Dot Score: {fs(gold_dot)}, MLP Score: {fs(gold_mlp)}")
        print(f"  Pred candidate -> Dot Score: {fs(pred_dot)}, MLP Score: {fs(pred_mlp)}")
        print("-" * 50)

if __name__ == "__main__":
    main()
