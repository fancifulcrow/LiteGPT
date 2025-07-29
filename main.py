from modules.dataset import TextDataset, split_dataset
from modules.model import LiteGPT
from modules.train import train_gpt
from modules.eval import generate_text, evaluate
from modules.utils import count_parameters, loss_curve, load_configuration

import argparse
import torch
from torch.utils.data import DataLoader
import tiktoken
import os
import math


def main() -> None:
    parser = argparse.ArgumentParser(description='Training and Inference of LiteGPT')
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file (default: config/default.yaml)")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "evaluate"], default="train", help="Train model, run inference, or evaluate performance")
    parser.add_argument("--prompt", type=str, default=" ", help="Enter your prompt")
    parser.add_argument("--weights", type=str, help="Path to the model weights in .pth file")
    args = parser.parse_args()

    config = load_configuration(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    context_length = config["model"]["context_length"]
    d_model = config["model"]["d_model"]
    num_heads = config["model"]["num_heads"]
    num_layers = config["model"]["num_layers"]
    ff_dim = config["model"]["ff_dim"]
    dropout = config["model"]["dropout"]

    data_path = config["data"]["path"]
    stride = config["data"]["stride"]
    num_workers = config["data"]["num_workers"]

    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]

    dataset = TextDataset(
        tokenizer=tokenizer, 
        context_length=context_length,
        stride=stride,
        folder_path=data_path
    )

    train_dataset, test_dataset = split_dataset(dataset, train_size=0.8, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = LiteGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Total Number of Parameters: {count_parameters(model)}")

    losses = train_gpt(model, optimizer, train_dataloader, num_epochs, device)

    loss_curve(losses, title="Training Loss")

    prompt = args.prompt
    generated_text = generate_text(model, tokenizer, prompt)

    print(generated_text)

    os.makedirs("models", exist_ok=True)
    model_save_path = "models/litegpt_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    test_loss, top_k_acc = evaluate(model, test_dataloader, device)

    print(f"Test Loss: {test_loss}")
    print(f"Top-5 Accuracy: {top_k_acc * 100:.4f}%")
    print(f"Perplexity: {math.exp(test_loss)}") # Perplexity = e^{cross_entropy_loss}


if __name__ == "__main__":
    main()
