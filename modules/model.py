import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteGPT(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, embedding_dim=256, num_heads=8, num_layers=4, ff_dim=512, dropout=0.1) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embedding_dim, bias=False)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, input_ids:torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
    def generate(self, input_ids:torch.Tensor, max_new_tokens=50, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, -float('Inf')).scatter_(1, indices, values)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
