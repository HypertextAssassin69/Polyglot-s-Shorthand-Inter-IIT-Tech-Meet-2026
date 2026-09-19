import json
import torch
import random
from torch.utils.data import Dataset
import re

class V1Dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, split="train", seed=42):
        self.tokenizer = tokenizer
        
        triplets = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    triplets.append(json.loads(line))
                    
        unique_sentences = list(set([t["raw_sentence"] for t in triplets]))
        unique_sentences.sort() 
        
        random.seed(seed)
        random.shuffle(unique_sentences)
        
        n = len(unique_sentences)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        train_sents = set(unique_sentences[:train_end])
        val_sents = set(unique_sentences[train_end:val_end])
        test_sents = set(unique_sentences[val_end:])
        
        if split == "train":
            target_sents = train_sents
        elif split == "val":
            target_sents = val_sents
        else:
            target_sents = test_sents
            
        self.data = [t for t in triplets if t["raw_sentence"] in target_sents]
        
        self.char2id = {"<pad>": 0, "<unk>": 1}
        for t in triplets:
            for cand in t["candidates"]:
                for char in cand:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)
        
        self.char_vocab_size = len(self.char2id)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["raw_sentence"]
        noisy_word = item["noisy_word"]
        target = item["target_word"]
        candidates = item["candidates"]
        
        encoded = self.tokenizer.encode(sentence)
        input_ids = encoded.ids
        offsets = encoded.offsets
        
        match = re.search(r'\b' + re.escape(noisy_word) + r'\b', sentence)
        if match:
            start_char, end_char = match.span()
        else:
            start_char = sentence.find(noisy_word)
            end_char = start_char + len(noisy_word)
            
        target_token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            overlap_start = max(start_char, tok_start)
            overlap_end = min(end_char, tok_end)
            if overlap_end > overlap_start:
                target_token_indices.append(i)
                
        try:
            target_idx = candidates.index(target)
        except ValueError:
            target_idx = -1 
            
        cand_char_ids = []
        for cand in candidates:
            c_ids = [self.char2id.get(c, self.char2id["<unk>"]) for c in cand]
            cand_char_ids.append(c_ids)
            
        return {
            "input_ids": input_ids,
            "target_token_indices": target_token_indices,
            "cand_char_ids": cand_char_ids,
            "target_idx": target_idx,
            "raw_sentence": sentence,
            "noisy_word": noisy_word,
            "target_word": target,
            "candidates": candidates
        }

def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    target_token_indices = [b["target_token_indices"] for b in batch]
    target_idx = torch.tensor([b["target_idx"] for b in batch], dtype=torch.long)
    
    max_word_len = max([len(c) for b in batch for c in b["cand_char_ids"]])
    max_cands = max([len(b["cand_char_ids"]) for b in batch])
    
    B = len(batch)
    char_ids = torch.zeros((B, max_cands, max_word_len), dtype=torch.long)
    candidate_mask = torch.zeros((B, max_cands), dtype=torch.bool)
    
    for i, b in enumerate(batch):
        cands = b["cand_char_ids"]
        for j, cand in enumerate(cands):
            char_ids[i, j, :len(cand)] = torch.tensor(cand)
            candidate_mask[i, j] = True
            
    raw_sentences = [b["raw_sentence"] for b in batch]
    noisy_words = [b["noisy_word"] for b in batch]
    target_words = [b["target_word"] for b in batch]
    candidates = [b["candidates"] for b in batch]
            
    return {
        "input_ids": input_ids,
        "target_spans": target_token_indices,
        "char_ids": char_ids,
        "candidate_mask": candidate_mask,
        "target_idx": target_idx,
        "raw_sentences": raw_sentences,
        "noisy_words": noisy_words,
        "target_words": target_words,
        "candidates": candidates
    }
