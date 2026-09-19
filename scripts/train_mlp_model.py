import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import time
import json

sys.path.append(os.getcwd())
from src.polyglot.models.v1_normalization import BiEncoderRanker, BiEncoderRankerMLP
from src.polyglot.data.v1_dataset import V1Dataset, collate_fn
from tests.phonetic_benchmark import edit_distance

DATA_PATH = "data/processed/v1_hybrid_triplets.jsonl"
TOK_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/tokenizer.json"
PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/mlp_predictions.jsonl"
MODEL_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/mlp_model.pt"

def sanity_check(model, dl, device):
    print("--- Running Sanity Check ---")
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    batch = next(iter(dl))
    input_ids = batch["input_ids"].to(device)
    char_ids = batch["char_ids"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    target_idx = batch["target_idx"].to(device)
    target_spans = batch["target_spans"]
    
    initial_loss = None
    final_loss = None
    initial_weight = model.scorer[0].weight.data.clone()
    for i in range(10):
        optimizer.zero_grad()
        logits = model(input_ids, target_spans, char_ids, candidate_mask)
        loss = criterion(logits, target_idx)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        if i == 9:
            final_loss = loss.item()
            
    final_weight = model.scorer[0].weight.data.clone()
    diff = (final_weight - initial_weight).abs().sum().item()
    print(f"MLP Linear(512, 256) weight diff: {diff:.6f}")
    
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss (10 iters): {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during sanity check!"
    assert diff > 0.0, "MLP parameters did not update!"
    print("Sanity Check Passed. Gradients are flowing.\n")
    
    return True

def evaluate(model, dl, device, split_name="Val", save_preds=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    
    total_loss = 0.0
    total = 0
    correct = 0
    
    gold_present_total = 0
    gold_present_correct = 0
    
    gold_absent_total = 0
    
    subset_non_min_ed_total = 0
    subset_non_min_ed_correct = 0
    subset_tie_total = 0
    subset_tie_correct = 0
    
    vowel_keywords = {"mama", "mamma", "kam", "kaam", "sath", "saath", "pani", "paani", "muje", "mujhe", "kese", "kaise"}
    subset_minimal_total = 0
    subset_minimal_correct = 0
    
    cand_counts = []
    predictions = []
    total_time = 0
    valid_loss_items = 0
    
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            target_idx = batch["target_idx"].to(device)
            target_spans = batch["target_spans"]
            
            raw_sentences = batch["raw_sentences"]
            noisy_words = batch["noisy_words"]
            target_words = batch["target_words"]
            candidates = batch["candidates"]
            
            start_time = time.time()
            if save_preds:
                logits, dot_logits = model(input_ids, target_spans, char_ids, candidate_mask, return_dot_product=True)
            else:
                logits = model(input_ids, target_spans, char_ids, candidate_mask)
            preds = logits.argmax(dim=-1).tolist()
            total_time += time.time() - start_time
            
            loss = criterion(logits, target_idx)
            total_loss += loss.item()
            valid_loss_items += (target_idx != -1).sum().item()
            
            target_idx_list = target_idx.tolist()
            
            for i in range(len(preds)):
                total += 1
                cands = candidates[i]
                cand_counts.append(len(cands))
                noisy = noisy_words[i]
                target = target_words[i]
                t_idx = target_idx_list[i]
                
                if t_idx == -1:
                    gold_absent_total += 1
                    is_correct = False
                else:
                    gold_present_total += 1
                    is_correct = (preds[i] == t_idx)
                    if is_correct:
                        correct += 1
                        gold_present_correct += 1
                        
                # Subset stats
                filtered_cands = [c for c in cands if c != noisy]
                if filtered_cands and t_idx != -1:
                    scored_cands = [(edit_distance(noisy, c), c) for c in filtered_cands]
                    scored_cands.sort()
                    min_ed = scored_cands[0][0]
                    target_ed = edit_distance(noisy, target)
                    
                    if target_ed > min_ed:
                        subset_non_min_ed_total += 1
                        if is_correct: subset_non_min_ed_correct += 1
                    elif target_ed == min_ed:
                        tied_cands = [c for ed, c in scored_cands if ed == min_ed]
                        if len(tied_cands) > 1 and target in tied_cands:
                            subset_tie_total += 1
                            if is_correct: subset_tie_correct += 1
                                
                is_minimal = False
                if noisy in vowel_keywords or target in vowel_keywords:
                    is_minimal = True
                else:
                    for c in cands:
                        if c in vowel_keywords:
                            is_minimal = True
                            break
                if is_minimal and t_idx != -1:
                    subset_minimal_total += 1
                    if is_correct: subset_minimal_correct += 1
                        
                if save_preds:
                    pred_word = cands[preds[i]] if preds[i] < len(cands) else "<pad>"
                    predictions.append({
                        "raw_sentence": raw_sentences[i],
                        "noisy_word": noisy,
                        "target_word": target,
                        "candidates": cands,
                        "predicted_idx": preds[i],
                        "target_idx": t_idx,
                        "predicted_word": pred_word,
                        "is_correct": is_correct,
                        "error_type": "None" if is_correct else ("Retriever Failure" if t_idx == -1 else "Ranker Failure"),
                        "mlp_score_pred": logits[i, preds[i]].item(),
                        "mlp_score_gold": logits[i, t_idx].item() if t_idx != -1 else None,
                        "dot_score_pred": dot_logits[i, preds[i]].item(),
                        "dot_score_gold": dot_logits[i, t_idx].item() if t_idx != -1 else None
                    })
                    
    avg_loss = total_loss / valid_loss_items if valid_loss_items > 0 else 0
    recall_1 = correct / total if total > 0 else 0
    
    print(f"[{split_name}] Loss: {avg_loss:.4f} | Recall@1: {recall_1:.2%} ({correct}/{total})")
    
    if split_name == "Test":
        cand_counts.sort()
        print(f"\n--- Test Set Candidate Stats ---")
        print(f"Mean Cands: {sum(cand_counts)/len(cand_counts):.2f}")
        print(f"Median Cands: {cand_counts[len(cand_counts)//2]}")
        print(f"P95 Cands: {cand_counts[int(len(cand_counts)*0.95)]}")
        
        print(f"\n--- Detailed Breakdown ---")
        print(f"Total Errors: {total - correct}")
        print(f"Retriever Failure Rate (Gold Absent): {gold_absent_total/total:.2%} ({gold_absent_total})")
        print(f"Ranker Failure Rate (Gold Present but Wrong): {(gold_present_total - gold_present_correct)/total:.2%} ({gold_present_total - gold_present_correct})")
        print(f"Recall@1 given Gold IS in Candidates: {gold_present_correct/gold_present_total:.2%}")
        print(f"Recall@1 given Gold NOT in Candidates: 0.00%")
        
        print(f"\n--- Subset Preservations ---")
        print(f"Non-Min ED Subset: {subset_non_min_ed_correct}/{subset_non_min_ed_total} ({subset_non_min_ed_correct/subset_non_min_ed_total:.2%} if > 0)")
        print(f"ED Tie Subset: {subset_tie_correct}/{subset_tie_total} ({subset_tie_correct/subset_tie_total:.2%} if > 0)")
        print(f"Minimal Pair Subset: {subset_minimal_correct}/{subset_minimal_total} ({subset_minimal_correct/subset_minimal_total:.2%} if > 0)")
        print(f"\nInference Latency: {(total_time / total) * 1000:.2f} ms/word")
        
        if save_preds:
            with open(PRED_PATH, 'w') as f:
                for p in predictions:
                    f.write(json.dumps(p) + '\n')
            print(f"\nPredictions saved to {PRED_PATH}")
            
    return avg_loss, recall_1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    tokenizer = Tokenizer.from_file(TOK_PATH)
    vocab_size = tokenizer.get_vocab_size()
    
    train_ds = V1Dataset(DATA_PATH, tokenizer, split="train", seed=42)
    val_ds = V1Dataset(DATA_PATH, tokenizer, split="val", seed=42)
    test_ds = V1Dataset(DATA_PATH, tokenizer, split="test", seed=42)
    
    char_vocab_size = train_ds.char_vocab_size
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 1. Sanity Check
    model = BiEncoderRankerMLP(vocab_size=vocab_size, char_vocab_size=char_vocab_size).to(device)
    
    ctx_params = sum(p.numel() for p in model.context_encoder.parameters() if p.requires_grad)
    cand_params = sum(p.numel() for p in model.candidate_encoder.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in model.scorer.parameters() if p.requires_grad)
    print(f"Context Encoder Params: {ctx_params:,}")
    print(f"Candidate Encoder Params: {cand_params:,}")
    print(f"MLP Scorer Params: {mlp_params:,}")
    print(f"Total Params: {ctx_params + cand_params + mlp_params:,}\n")
    
    sanity_check(model, train_dl, device)
    
    # Re-initialize for real training
    model = BiEncoderRankerMLP(vocab_size=vocab_size, char_vocab_size=char_vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    epochs = 15
    best_val_loss = float('inf')
    
    print("--- Starting Full Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_items = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            target_idx = batch["target_idx"].to(device)
            target_spans = batch["target_spans"]
            
            optimizer.zero_grad()
            logits = model(input_ids, target_spans, char_ids, candidate_mask)
            loss = criterion(logits, target_idx)
            
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (target_idx != -1).sum().item()
                valid_items += (target_idx != -1).sum().item()
            
        avg_train_loss = total_loss / valid_items if valid_items > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        val_loss, val_acc = evaluate(model, val_dl, device, split_name="Val")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("  * Best validation loss achieved. Model saved.")
            
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    evaluate(model, test_dl, device, split_name="Test", save_preds=True)
    
if __name__ == "__main__":
    train()
