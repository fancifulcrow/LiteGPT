import torch
import torch.nn.functional as F
from tqdm import tqdm

def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=1.0, top_k=50, device=None) -> str:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids[:, -model.context_length:]
            logits = model(input_ids_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('Inf')).scatter_(1, indices, values)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())

    return generated_text


def top_k_accuracy(output, target, k=5) -> float:
    topk = torch.topk(output, k=k, dim=1).indices
    correct = topk.eq(target.view(-1, 1).expand_as(topk))
    correct_total = correct.sum().item()
    accuracy = correct_total / target.size(0)
    
    return accuracy


def evaluate(model, criterion, dataloader, device, k=5) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_top_k_acc = 0.0
    total_batches = 0

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
