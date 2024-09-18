from modules.data import TextDataset, load_data
from modules.model import LiteGPT
from modules.train import train
from modules.utils import count_parameters, generate_text

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tiktoken
import yaml


config_file_path = "config/config.yaml"

with open(config_file_path, mode="r") as f:
    config = yaml.safe_load(f)


def main() -> None:
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

    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)

    model = LiteGPT(
        vocab_size=vocab_size,
        context_length=config["model"]["context_length"],
        embedding_dim=config["model"]["embedding_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        ff_dim=config["model"]["ff_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Total Number of Parameters: {count_parameters(model)}")

    losses = train(model, optimizer, criterion, dataloader, config["training"]["num_epochs"], device)

    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    prompt = "I believe that "
    generated_text = generate_text(model, tokenizer, prompt, temperature=0.8)

    print(generated_text)

    model_save_path = "models/litegpt_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()