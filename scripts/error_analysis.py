import json
import random

def main():
    random.seed(42)
    with open("scratch/byt5_errors.json", "r", encoding="utf-8") as f:
        errors = json.load(f)
        
    sample = random.sample(errors, min(50, len(errors)))
    
    with open("scratch/error_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
        
    print(f"Sampled {len(sample)} errors to scratch/error_sample.json")

if __name__ == "__main__":
    main()
