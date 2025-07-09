import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteGPT(nn.Module):
    def __init__(
        self, 
        vocab_size:int, 
        context_length:int, 
        d_model: int = 256, 
        num_heads: int = 8, 
        num_layers: int = 4, 
        ff_dim: int = 512, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )
        
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1).to(input_ids.device)
        x = self.transformer_encoder(x, mask=causal_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            input_crop = input_ids[:, -self.context_length:]

            logits = self.forward(input_crop)
            next_token_logits = logits[:, -1, :]

            next_token_logits /= temperature

            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
