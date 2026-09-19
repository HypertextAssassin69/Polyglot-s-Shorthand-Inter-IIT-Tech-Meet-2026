import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    def __init__(self, vocab_size=131072, d_model=128, nhead=2, num_layers=2, dim_feedforward=512, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, input_ids):
        # input_ids: [B, S]
        B, S = input_ids.size()
        
        # Clamp S to max_seq_len just in case (though we assume <=128)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.emb_norm(x)
        
        mask = (input_ids == 0) 
        
        out = self.transformer(x, src_key_padding_mask=mask)
        return out # [B, S, d_model]

class CandidateEncoder(nn.Module):
    def __init__(self, char_vocab_size=50, char_dim=32, out_channels=128, kernel_size=3):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim)
        # padding=1 ensures sequence length remains the same
        self.conv = nn.Conv1d(in_channels=char_dim, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        
    def forward(self, char_ids):
        # char_ids: [B, num_candidates, max_word_len]
        B, K, W = char_ids.size()
        
        # Flatten B and K to process all candidates concurrently
        x = char_ids.view(B * K, W)
        
        # Embed chars
        x = self.char_embedding(x) # [B*K, W, char_dim]
        
        # Conv1D expects [Batch, Channels, Length]
        x = x.transpose(1, 2) # [B*K, char_dim, W]
        
        x = self.conv(x) # [B*K, out_channels, W]
        x = F.relu(x)
        
        # Global Max Pooling over length dimension (dim=2)
        x, _ = torch.max(x, dim=2) # [B*K, out_channels]
        
        # Reshape back to [B, K, out_channels]
        out = x.view(B, K, -1)
        return out

class BiEncoderRanker(nn.Module):
    def __init__(self, vocab_size=131072, char_vocab_size=50):
        super().__init__()
        self.context_encoder = ContextEncoder(vocab_size=vocab_size)
        self.candidate_encoder = CandidateEncoder(char_vocab_size=char_vocab_size)
        
    def forward(self, input_ids, target_spans, char_ids, candidate_mask):
        B = input_ids.size(0)
        
        # 1. Encode Context
        context_out = self.context_encoder(input_ids) # [B, S, 128]
        
        # Extract and mean-pool target spans
        context_vectors = []
        for b in range(B):
            span_indices = target_spans[b]
            if len(span_indices) == 0:
                span_vec = torch.zeros(128, device=context_out.device)
            else:
                span_vec = context_out[b, span_indices].mean(dim=0) # [128]
            context_vectors.append(span_vec)
            
        context_vector = torch.stack(context_vectors) # [B, 128]
        
        # 2. Encode Candidates
        candidate_vectors = self.candidate_encoder(char_ids) # [B, K, 128]
        
        # 3. Score (Dot Product)
        # context_vector: [B, 1, 128]
        # candidate_vectors: [B, 128, K]
        # bmm gives: [B, 1, K]
        scores = torch.bmm(context_vector.unsqueeze(1), candidate_vectors.transpose(1, 2)).squeeze(1) # [B, K]
        
        # 4. Mask invalid candidates
        # Set scores of invalid candidates (padding) to -inf
        scores = scores.masked_fill(~candidate_mask, float('-inf'))
        
        return scores

class BiEncoderRankerMLP(nn.Module):
    def __init__(self, vocab_size=131072, char_vocab_size=50):
        super().__init__()
        self.context_encoder = ContextEncoder(vocab_size=vocab_size)
        self.candidate_encoder = CandidateEncoder(char_vocab_size=char_vocab_size)
        
        self.scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, target_spans, char_ids, candidate_mask, return_dot_product=False):
        B = input_ids.size(0)
        
        # 1. Encode Context
        context_out = self.context_encoder(input_ids) # [B, S, 128]
        
        # Extract and mean-pool target spans
        context_vectors = []
        for b in range(B):
            span_indices = target_spans[b]
            if len(span_indices) == 0:
                span_vec = torch.zeros(128, device=context_out.device)
            else:
                span_vec = context_out[b, span_indices].mean(dim=0) # [128]
            context_vectors.append(span_vec)
            
        context_vector = torch.stack(context_vectors) # [B, 128]
        
        # 2. Encode Candidates
        candidate_vectors = self.candidate_encoder(char_ids) # [B, K, 128]
        
        # 3. Score (MLP)
        K = candidate_vectors.size(1)
        # Broadcast context_vector to [B, K, 128]
        c = context_vector.unsqueeze(1).expand(B, K, 128)
        x = candidate_vectors
        
        diff = torch.abs(c - x)
        mult = c * x
        
        interaction = torch.cat([c, x, diff, mult], dim=-1) # [B, K, 512]
        scores = self.scorer(interaction).squeeze(-1) # [B, K]
        
        # 4. Mask invalid candidates
        scores = scores.masked_fill(~candidate_mask, float('-inf'))
        
        if return_dot_product:
            dot_scores = torch.bmm(context_vector.unsqueeze(1), candidate_vectors.transpose(1, 2)).squeeze(1)
            dot_scores = dot_scores.masked_fill(~candidate_mask, float('-inf'))
            return scores, dot_scores
            
        return scores

class CrossEncoderRanker(nn.Module):
    def __init__(self, vocab_size=131072, char_vocab_size=50):
        super().__init__()
        # Token and Character Embeddings mapped to the same dimension
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.char_embedding = nn.Embedding(char_vocab_size, 128)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.sep_token = nn.Parameter(torch.randn(1, 1, 128))
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(256, 128)
        self.emb_norm = nn.LayerNorm(128)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Scorer
        self.scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, target_spans, char_ids, candidate_mask):
        B, K, C = char_ids.shape
        S = input_ids.size(1)
        
        # Context embeddings: [B, S, 128]
        ctx_emb = self.token_embedding(input_ids)
        # Expand context to [B, K, S, 128]
        ctx_emb = ctx_emb.unsqueeze(1).expand(B, K, S, 128)
        
        # Candidate embeddings: [B, K, C, 128]
        cand_emb = self.char_embedding(char_ids)
        
        # Expand CLS and SEP to [B, K, 1, 128]
        cls_emb = self.cls_token.expand(B, K, 1, 128)
        sep_emb = self.sep_token.expand(B, K, 1, 128)
        
        # Concatenate: [CLS] context [SEP] candidate [SEP]
        # Shape: [B, K, 1 + S + 1 + C + 1, 128]
        seq_emb = torch.cat([cls_emb, ctx_emb, sep_emb, cand_emb, sep_emb], dim=2)
        SeqLen = seq_emb.size(2)
        
        # Add positional embedding
        positions = torch.arange(SeqLen, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(B, K, SeqLen)
        seq_emb = seq_emb + self.pos_embedding(positions)
        seq_emb = self.emb_norm(seq_emb)
        
        # Padding masks (True where padding)
        # Context padding is 0. Expand to [B, K, S]
        ctx_pad_mask = (input_ids == 0).unsqueeze(1).expand(B, K, S)
        
        # Candidate padding is 0
        cand_pad_mask = (char_ids == 0) # [B, K, C]
        
        # Special tokens are never padded
        cls_mask = torch.zeros(B, K, 1, dtype=torch.bool, device=input_ids.device)
        sep_mask = torch.zeros(B, K, 1, dtype=torch.bool, device=input_ids.device)
        
        # Combined mask: [B, K, 1 + S + 1 + C + 1]
        pad_mask = torch.cat([cls_mask, ctx_pad_mask, sep_mask, cand_pad_mask, sep_mask], dim=2)
        
        # Flatten batch and candidate dims to pass through Transformer
        seq_emb = seq_emb.view(B * K, -1, 128)
        pad_mask = pad_mask.view(B * K, -1)
        
        # Transformer forward pass
        out = self.transformer(seq_emb, src_key_padding_mask=pad_mask) # [B*K, SeqLen, 128]
        
        # Extract [CLS] embedding
        cls_out = out[:, 0, :] # [B*K, 128]
        
        # Score
        scores = self.scorer(cls_out) # [B*K, 1]
        scores = scores.view(B, K) # [B, K]
        
        # Mask invalid candidates
        scores = scores.masked_fill(~candidate_mask, float('-inf'))
        
        return scores

