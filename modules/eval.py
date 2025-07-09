import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple

from .models import LiteGPT


def generate_text(
        model: LiteGPT, 
        tokenizer, 
        prompt, 
        max_new_tokens: int = 128, 
        temperature: float = 1.0, 
        top_k: int = 0, 
        device: Optional[str] = None
    ) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    # Encode using tiktoken
    input_ids = tokenizer.encode(prompt) # (seq_len)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)  # (1, seq_len)

    input_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    # Decode using tiktoken
    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int) -> float:
    topk = torch.topk(output, k=k, dim=1).indices
    correct = topk.eq(target.view(-1, 1).expand_as(topk))
    correct_total = correct.sum().item()
    accuracy = correct_total / target.size(0)
    
    return accuracy


def evaluate(model: LiteGPT, dataloader: torch.utils.data.DataLoader, device: Optional[str], k: int = 5) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_top_k_acc = 0.0
    total_batches = 0

    criterion = nn.CrossEntropyLoss()

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for _, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            running_loss += loss.item()

            batch_size, seq_length, num_classes = logits.size()

            top_k_acc = top_k_accuracy(
                logits.view(batch_size * seq_length, num_classes),
                target_ids.view(batch_size * seq_length), 
                k=k
            )

            running_top_k_acc += top_k_acc

            total_batches += 1

            progress_bar.set_postfix({
                "loss": running_loss / total_batches, 
                f"top-{k}-acc": running_top_k_acc / total_batches
            })

    average_loss = running_loss / total_batches
    average_top_k_accuracy = running_top_k_acc / total_batches

    return average_loss, average_top_k_accuracy
