from modules.dataset import TextDataset, split_dataset
from modules.model import LiteGPT
from modules.train import train_gpt
from modules.eval import generate_text, evaluate_gpt
from modules.utils import count_parameters, loss_curve, load_configuration

import argparse
import torch
from torch.utils.data import DataLoader
import tiktoken
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description='Training and Inference of LiteGPT')
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file (default: config/default.yaml)")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "evaluate"], default="train", help="Train model, run inference, or evaluate performance")
    parser.add_argument("--prompt", type=str, default=" ", help="Enter your prompt")
    parser.add_argument("--weights", type=str, help="Path to the model weights in .pth file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("file.log", mode="w")
        ]
    )

    logging.info(f"Running in {args.mode.upper()} mode")

    logging.info(f"Command-line arguments: {vars(args)}")
    config = load_configuration(args.config)
    logging.info(f"Configuration:\n {config}")
    
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
    use_scheduler = config["training"]["use_scheduler"]

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

    is_trained = False

    if args.weights:
        is_trained = True
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded pretrained LiteGPT weights from {args.weights}")
        logging.info(f"Loaded pretrained LiteGPT weights from {args.weights}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) if use_scheduler else None

    print(f"Total Number of Parameters: {count_parameters(model)}")
    logging.info(f"Total Number of Parameters: {count_parameters(model)}")

    # Train
    if args.mode in {"train"}:
        losses = train_gpt(model, optimizer, train_dataloader, num_epochs, device, scheduler=scheduler)
        loss_curve(losses, title="Training Loss")
        is_trained = True

    # Evaluate
    if args.mode in {"train", "evaluate"}:
        if not is_trained:
            raise ValueError("Evaluation requires a pretrained model. Specify with --weights or train from scratch")

        test_loss, top_k_acc, perplexity = evaluate_gpt(model, test_dataloader, device)

        print(f"Test Loss: {test_loss}")
        print(f"Top-5 Accuracy: {top_k_acc * 100:.4f}%")
        print(f"Perplexity: {perplexity}")

    # Inference
    if args.mode in {"train", "evaluate", "inference"}:
        if not is_trained:
            raise ValueError("Inference requires a pretrained model. Specify with --weights or train from scratch")

        prompt = args.prompt
        generated_text = generate_text(model, tokenizer, prompt)

        print(generated_text)


if __name__ == "__main__":
    main()
