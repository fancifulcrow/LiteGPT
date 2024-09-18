from tqdm import tqdm

def train(model, optimizer, criterion, dataloader, num_epochs:int, device) -> list[float]:
    model.train()

    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            
            progress_bar.set_postfix({"loss": running_loss / (batch_idx + 1)})

    return losses