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
from src.polyglot.models.v1_normalization import BiEncoderRanker
from src.polyglot.data.v1_dataset import V1Dataset, collate_fn
from tests.phonetic_benchmark import edit_distance

DATA_PATH = "data/processed/v1_candidate_ranking_triplets.jsonl"
TOK_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/tokenizer.json"
PRED_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/v1_predictions.jsonl"
MODEL_PATH = "C:/Users/hiaar/.gemini/antigravity/brain/2b8c0d59-79d1-442a-98a8-9a8f70fc8e10/scratch/v1_model.pt"

def sanity_check(model, dl, device):
    print("--- Running Sanity Check ---")
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    batch = next(iter(dl))
    input_ids = batch["input_ids"].to(device)
    char_ids = batch["char_ids"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    target_idx = batch["target_idx"].to(device)
    target_spans = batch["target_spans"]
    
    initial_loss = None
    final_loss = None
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
            
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss (10 iters): {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during sanity check!"
    print("Sanity Check Passed. Gradients are flowing.\n")
    
    # Reset model weights for full training (lazy reset by recreating)
    return True

def evaluate(model, dl, device, split_name="Val", save_preds=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    total_loss = 0.0
    total = 0
    correct = 0
    
    subset_non_min_ed_total = 0
    subset_non_min_ed_correct = 0
    
    subset_tie_total = 0
    subset_tie_correct = 0
    
    vowel_keywords = {"mama", "mamma", "kam", "kaam", "sath", "saath", "pani", "paani", "muje", "mujhe", "kese", "kaise"}
    subset_minimal_total = 0
    subset_minimal_correct = 0
    
    predictions = []
    total_time = 0
    
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
            logits = model(input_ids, target_spans, char_ids, candidate_mask)
            preds = logits.argmax(dim=-1).tolist()
            total_time += time.time() - start_time
            
            loss = criterion(logits, target_idx)
            total_loss += loss.item()
            
            target_idx_list = target_idx.tolist()
            
            for i in range(len(preds)):
                total += 1
                is_correct = (preds[i] == target_idx_list[i])
                if is_correct:
                    correct += 1
                    
                noisy = noisy_words[i]
                target = target_words[i]
                cands = candidates[i]
                
                # Exclude identity candidate for baseline checks
                filtered_cands = [c for c in cands if c != noisy]
                if filtered_cands:
                    scored_cands = [(edit_distance(noisy, c), c) for c in filtered_cands]
                    scored_cands.sort()
                    min_ed = scored_cands[0][0]
                    target_ed = edit_distance(noisy, target)
                    
                    if target_ed > min_ed:
                        subset_non_min_ed_total += 1
                        if is_correct:
                            subset_non_min_ed_correct += 1
                    elif target_ed == min_ed:
                        tied_cands = [c for ed, c in scored_cands if ed == min_ed]
                        if len(tied_cands) > 1 and target in tied_cands:
                            subset_tie_total += 1
                            if is_correct:
                                subset_tie_correct += 1
                                
                is_minimal = False
                if noisy in vowel_keywords or target in vowel_keywords:
                    is_minimal = True
                else:
                    for c in cands:
                        if c in vowel_keywords:
                            is_minimal = True
                            break
                            
                if is_minimal:
                    subset_minimal_total += 1
                    if is_correct:
                        subset_minimal_correct += 1
                        
                if save_preds:
                    predictions.append({
                        "raw_sentence": raw_sentences[i],
                        "noisy_word": noisy,
                        "target_word": target,
                        "candidates": cands,
                        "predicted_idx": preds[i],
                        "predicted_word": cands[preds[i]],
                        "is_correct": is_correct
                    })
                    
    avg_loss = total_loss / total
    recall_1 = correct / total if total > 0 else 0
    
    print(f"[{split_name}] Loss: {avg_loss:.4f} | Recall@1: {recall_1:.2%} ({correct}/{total})")
    
    if split_name == "Test":
        print(f"  Non-Min ED Subset: {subset_non_min_ed_correct}/{subset_non_min_ed_total} ({subset_non_min_ed_correct/subset_non_min_ed_total:.2%} if > 0)")
        print(f"  ED Tie Subset: {subset_tie_correct}/{subset_tie_total} ({subset_tie_correct/subset_tie_total:.2%} if > 0)")
        print(f"  Minimal Pair Subset: {subset_minimal_correct}/{subset_minimal_total} ({subset_minimal_correct/subset_minimal_total:.2%} if > 0)")
        print(f"  Inference Latency: {(total_time / total) * 1000:.2f} ms/word")
        
        if save_preds:
            with open(PRED_PATH, 'w') as f:
                for p in predictions:
                    f.write(json.dumps(p) + '\n')
            print(f"Predictions saved to {PRED_PATH}")
            
            # Print 5 error examples
            print("\nRepresentative Test Errors:")
            errors = [p for p in predictions if not p["is_correct"]]
            for p in errors[:5]:
                print(f"Noisy: {p['noisy_word']:<10} Target: {p['target_word']:<10} Pred: {p['predicted_word']:<10} Cands: {p['candidates']}")
                
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
    model = BiEncoderRanker(vocab_size=vocab_size, char_vocab_size=char_vocab_size).to(device)
    sanity_check(model, train_dl, device)
    
    # Re-initialize for real training
    model = BiEncoderRanker(vocab_size=vocab_size, char_vocab_size=char_vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    best_val_loss = float('inf')
    
    print("--- Starting Full Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            target_idx = batch["target_idx"].to(device)
            target_spans = batch["target_spans"]
            
            optimizer.zero_grad()
            logits = model(input_ids, target_spans, char_ids, candidate_mask)
            loss = criterion(logits, target_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dl)
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
