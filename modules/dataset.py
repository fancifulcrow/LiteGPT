import torch
from torch.utils.data import Dataset, random_split
from typing import Tuple
import os


class TextDataset(Dataset):
    def __init__(self, tokenizer, context_length: int, stride: int, folder_path: str = "./data") -> None:
        self.context_length = context_length
        self.input_ids = []
        self.target_ids = []

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        with open(file_path, mode="r", encoding="utf-8") as f:
                            text = f.read()
                        
                        # Skip empty files
                        if not text:
                            continue
                            
                        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                        
                        # Need at least context_length + 1 tokens to create a sample
                        if len(tokens) < context_length + 1:
                            continue

                        # Create sliding windows
                        for i in range(0, len(tokens) - context_length, stride):
                            x = tokens[i:i + context_length]
                            y = tokens[i + 1:i + context_length + 1]
                            
                            # Ensure both sequences are exactly the right length
                            if len(x) == len(y) == context_length:
                                self.input_ids.append(torch.tensor(x, dtype=torch.long))
                                self.target_ids.append(torch.tensor(y, dtype=torch.long))
                    
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def split_dataset(dataset: TextDataset, train_size: float = 0.8, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    num_samples = len(dataset)
    train_length = int(train_size * num_samples)
    test_length = num_samples - train_length
    
    generator = torch.Generator().manual_seed(random_state)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_length, test_length],
        generator=generator
    )

    return train_dataset, test_dataset
