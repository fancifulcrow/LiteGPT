import torch
import torch.nn.functional as F

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


