import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW

def load_data(path, n=5):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    noisy_samples = [s for s in samples if s['is_noisy']]
    clean_samples = [s for s in samples if not s['is_noisy']]
    
    selected = noisy_samples[:n-1] + clean_samples[:1]
    return selected

def main():
    print("Loading data...")
    data = load_data("data/processed/byt5_normalization_triplets.jsonl", n=5)
    
    print("\n--- Example Data Format ---")
    for d in data[:2]:
        print(f"Input : {d['input_text']}")
        print(f"Target: {d['target_text']}")
        print(f"Noisy?: {d['is_noisy']}\n")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    print("\n--- Sanity Training (Overfitting 5 instances) ---")
    
    inputs = [d['input_text'] for d in data]
    targets = [d['target_text'] for d in data]
    
    encoded_inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(device)
    encoded_targets = tokenizer(targets, padding=True, return_tensors="pt").to(device)
    
    # Replace padding tokens with -100 for Cross Entropy Loss
    labels = encoded_targets.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(input_ids=encoded_inputs.input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
    print("\n--- Testing Generation After Sanity Train ---")
    model.eval()
    with torch.no_grad():
        gen_tokens = model.generate(input_ids=encoded_inputs.input_ids, max_new_tokens=20)
        
    for i, tokens in enumerate(gen_tokens):
        gen_text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Input : {inputs[i]}")
        print(f"Target: {targets[i]}")
        print(f"Output: {gen_text}")
        print(f"Match?: {gen_text.strip() == targets[i].strip()}\n")

if __name__ == "__main__":
    main()
