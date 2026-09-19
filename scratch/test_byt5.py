import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print("Loading ByT5-small...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    
    print(f"Loaded in {time.time() - t0:.2f} seconds.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Verify tokenizer handles markers
    text = "normalize: the dog <n> borked </n> loudly"
    encoded = tokenizer(text, return_tensors="pt")
    decoded = tokenizer.decode(encoded.input_ids[0])
    
    print(f"Input: {text}")
    print(f"Encoded shape: {encoded.input_ids.shape}")
    print(f"Decoded: {decoded}")
    
    print("Testing forward pass...")
    target = "barked"
    labels = tokenizer(target, return_tensors="pt").input_ids
    
    # Pass through model
    outputs = model(input_ids=encoded.input_ids, labels=labels)
    print(f"Loss: {outputs.loss.item()}")
    
    print("Testing generation pass...")
    gen_tokens = model.generate(input_ids=encoded.input_ids, max_new_tokens=20)
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"Generated text: {gen_text}")
    
if __name__ == "__main__":
    main()
