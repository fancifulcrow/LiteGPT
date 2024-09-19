from modules.data import TextDataset, load_data, split_dataset
from modules.model import LiteGPT
from modules.train import train
from modules.eval import generate_text, evaluate
from modules.utils import count_parameters, loss_curve, load_configuration

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken
import os
import math

import warnings


def main() -> None:
    warnings.filterwarnings("ignore")

    torch.manual_seed(42)

    config_path = "config/config.yaml"
    config = load_configuration(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_data(folder_path=config["data"]["path"])
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    dataset = TextDataset(
        text=data,
        tokenizer=tokenizer, 
        context_length=config["model"]["context_length"], 
        stride=config["data"]["stride"]
    )

    train_dataset, test_dataset = split_dataset(dataset, train_size=0.8)

    train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)

    model = LiteGPT(
        vocab_size=vocab_size,
        context_length=config["model"]["context_length"],
        embedding_dim=config["model"]["embedding_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        ff_dim=config["model"]["ff_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    print(f"Total Number of Parameters: {count_parameters(model)}")

    losses = train(model, optimizer, criterion, train_dataloader, config["training"]["num_epochs"], device)

    loss_curve(losses, title="Training Loss")

    prompt = "For years to come "
    generated_text = generate_text(model, tokenizer, prompt)

    print(generated_text)

    os.makedirs("models", exist_ok=True)
    model_save_path = "models/litegpt_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    test_loss, top_k_acc = evaluate(model, criterion, test_dataloader, device)

    print(f"Test Loss: {test_loss}")
    print(f"Top-5 Accuracy: {top_k_acc * 100:.4f}%")
    print(f"Perplexity: {math.exp(test_loss)}") # Perplexity = e^{cross_entropy_loss}


if __name__ == "__main__":
    main()
