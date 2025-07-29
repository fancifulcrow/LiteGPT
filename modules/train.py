import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional
import logging
import os

from .model import LiteGPT


def train_gpt(
        model: LiteGPT,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        device: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        checkpoint_dir: str = "checkpoints",
        save_dir: str = "models",
    ) -> List[float]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.train()

    epoch_losses = []
    criterion = nn.CrossEntropyLoss()

    logging.info("LiteGPT Training started")

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

            progress_bar.set_postfix({"loss": running_loss / (batch_idx + 1)})

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss}")

        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
    model_path = os.path.join(save_dir, "litegpt.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    logging.info("LiteGPT Training completed")

    return epoch_losses
