import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional

from .models import LiteGPT


def train_gpt(
    model: LiteGPT,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: Optional[str] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> List[float]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    losses = []
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

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

        if scheduler is not None:
            scheduler.step()

    return losses
