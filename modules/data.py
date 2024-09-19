import torch
from torch.utils.data import Dataset, random_split, Subset
import os


class TextDataset(Dataset):
    def __init__(self, text:str, tokenizer, context_length:int, stride:int) -> None:
        self.input_ids = []
        self.target_ids = []

        # Encode the text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Chunk text into overlapping sequences
        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx:int) ->tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def load_data(folder_path:str) -> str:
    text = ""

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith(".txt"):
            with open(file_path, mode="r", encoding="utf-8") as f:
                text += f.read() + "\n"

    return text


def split_dataset(dataset, train_size=0.8) -> tuple[Subset, Subset]:
    num_samples = len(dataset)
    train_length = int(train_size * num_samples)
    test_length = num_samples - train_length
    
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

    return train_dataset, test_dataset
