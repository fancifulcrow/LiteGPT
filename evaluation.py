from modules.model import LiteGPT
from modules.data import TextDataset, load_data, split_dataset
from modules.eval import evaluate, generate_text
from modules.utils import count_parameters, load_configuration

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken
import math
import warnings


def main() -> None:
    warnings.filterwarnings('ignore')
    torch.manual_seed(42)
    
    config_path = "config/config.yaml"
    config = load_configuration(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    data = load_data(folder_path=config["data"]["path"])

    dataset = TextDataset(
        text=data,
        tokenizer=tokenizer, 
        context_length=config["model"]["context_length"], 
        stride=config["data"]["stride"]
    )

    _, test_dataset = split_dataset(dataset, train_size=0.8)

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


    model_path = "models/litegpt_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Number of Parameters: {count_parameters(model)}")

    prompt = "Come to me "

    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=128)

    print(generated_text)

    loss, top_k_acc = evaluate(model, nn.CrossEntropyLoss(), test_dataloader, device, k=5)

    print(f"Test Loss: {loss:.4f}")
    print(f"Top-5 Accuracy: {top_k_acc * 100:.4f}%")
    print(f"Perplexity: {math.exp(loss)}") # Perplexity = e^{cross_entropy_loss}


if __name__ == "__main__":
    main()
