import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LiteGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
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

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Generates a causal mask with shape (seq_len, seq_len)
        return torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        positions = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = self._generate_causal_mask(seq_length, device)
        x = self.transformer_encoder(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 128, 
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_crop = input_ids[:, -self.context_length:]

                logits = self.forward(input_crop)
                next_token_logits = logits[:, -1, :]

                next_token_logits /= temperature

                # Top-k filtering
                if top_k is not None:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    threshold = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < threshold,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(
                        indices_to_remove,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
