import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import os
from tqdm import tqdm
from functools import partial

class NormalizationDataset(Dataset):
    def __init__(self, data_path, split="train"):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                # SMART FILTERING: Drop outliers longer than 150 characters to prevent O(N^2) explosion
                if d['split'] == split and len(d['input_text']) <= 150:
                    self.samples.append(d)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer):
    inputs = [item['input_text'] for item in batch]
    targets = [item['target_text'] for item in batch]
    
    # Max target length 30 since it's only generating single words
    encoded_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=150, return_tensors="pt")
    encoded_targets = tokenizer(targets, padding=True, truncation=True, max_length=30, return_tensors="pt")
    
    labels = encoded_targets.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': encoded_inputs.input_ids,
        'attention_mask': encoded_inputs.attention_mask,
        'labels': labels
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    model.to(device)
    
    # (Removed torch.compile since Triton is not natively supported on Windows)
    
    train_dataset = NormalizationDataset("data/processed/byt5_normalization_triplets.jsonl", "train")
    val_dataset = NormalizationDataset("data/processed/byt5_normalization_triplets.jsonl", "val")
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    batch_size = 16
    accum_steps = 2
    
    # num_workers=0 for Windows to avoid multiprocessing hangs
    collate_partial = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_partial,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_partial,
                            num_workers=0, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_model_path = "scratch/byt5_best_model.pt"
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        exact_match = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                # Check accuracy quickly during validation
                gen_tokens = model.generate(input_ids=input_ids, max_new_tokens=20)
                for j in range(len(gen_tokens)):
                    pred = tokenizer.decode(gen_tokens[j], skip_special_tokens=True).strip()
                    # Decode labels (ignore -100)
                    lbl = labels[j].clone()
                    lbl[lbl == -100] = tokenizer.pad_token_id
                    truth = tokenizer.decode(lbl, skip_special_tokens=True).strip()
                    if pred == truth:
                        exact_match += 1
                    total_val += 1
                    
        val_loss /= len(val_loader)
        val_acc = exact_match / total_val
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved new best model (Val Loss: {best_val_loss:.4f})")
            
    print("Training complete.")

if __name__ == "__main__":
    main()
