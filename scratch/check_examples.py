import json

def check_examples():
    data = []
    with open('data/processed/byt5_normalization_triplets.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    noisy = [d for d in data if d['is_noisy']]
    for d in noisy[:5]:
        print(f"Raw: {d['raw_sentence']}")
        print(f"ByT5: {d['input_text']}")
        print(f"Target: {d['target_text']}")
        print("-" * 50)

if __name__ == '__main__':
    check_examples()
