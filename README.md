# LiteGPT

LiteGPT is a minimal, flexible implementation of the GPT architecture built with PyTorch. It's designed for learning, experimentation, and lightweight use cases, making it ideal for those interested in the inner workings of transformer-based language models.

## Dataset
The model was been trained on a dataset featuring texts from the following public domain books, sourced from [Project Gutenberg](https://www.gutenberg.org/):

- *Frankenstein* by Mary Shelley
- *The Adventures of Sherlock Holmes* by Arthur Conan Doyle
- *Metamorphosis* by Franz Kafka
- *The Great Gatsby* by F. Scott Fitzgerald
- *Moby Dick* by Herman Melville

## Pre-Trained Model

Download the pretrained weights [here](https://github.com/fancifulcrow/LiteGPT/releases). 

With the default configuration, LiteGPT has **13 million** parameters. It utilizes the **GPT-2 tokenizer** from the `tiktoken` library.

## Installation

1. Clone the repository
    ```bash
    git clone https://github.com/fancifulcrow/LiteGPT.git
    cd LiteGPT
    ```

2. Install the required dependencies by running:
    ```bash
    pip install -r requirements
    ```
    Or
    ```bash
    pip install torch tiktoken numpy matplotlib pyyaml tqdm
    ```

## Configuration

All training and model parameters are defined in `config/default.yaml`. You can modify this file directly or supply a custom configuration using the `--config` flag when running the script.

## Usage

To run the main script, use:

```bash
python3 main.py
```

You can customize the behavior of the program using the following command-line arguments:
| Argument    | Description                                          |
| ----------- | ---------------------------------------------------- |
| `--config`  | Path to config file (default: `config/default.yaml`) |
| `--mode`    | `train`, `inference`, or `evaluate`                  |
| `--prompt`  | Prompt for text generation                           |
| `--weights` | Path to a `.pth` model file                          |

### Generated Samples

> And it is only when the elastic is an important-sided horn may be used in the shape of a large whale, and hangs down in his head-bone earrings: he only accidentally left board saw a cuttingist in a body. Of these three years of the White Whale, it is well known. He has long been for years been a small indeed peculiar fer, and is it not yet in general. He is the mere chance of seeing you, though a ferocity of the matter, in the matter, has not perceptible reluctance.



> “You have a good pawnbroker’s business.”
> 
> “What is our saying,” he remarked decisively.
> 
> “But the sailors had on an attack. “That’s a great difference between the affair and your mother’s in front of me.”
> 
> “Is this Miss Baker?”
> 
> “Yes.”
> 
> “You can’t understand it.”
> 
> “You can’t live in the matter,” said Wilson. “She’s only married …”
> 
> I insisted with him and I spoke of her strong emotion.


> —that’s inexorable!”
> 
> “Now,” said Gatsby, and with his expressive nose. “but it’s only a moment.”
> 
> “It’s only a moment.”
> 
> “A hard driver was not absolutely uncommunicable forever.’s when you will.”
> 
> “I’m paid a party”, old man, “or wait here and there unrestfully,” he said. “I’ll call up Daisy with me—”
> 
> The telephone rang inside—she said well with the sight into the house there, gently subsiding toward her beautiful and lovely tree accompanied her beautiful kitchen and strolled across the room. She was delighted to see that all the windows were sleeping in the dark gentle sky. The moon had long been accustomed to the sun, and a balust religions of the sun had risen higher, and a wildness in the morning, when a fire, slant, after the Parsee’s eyes came a burst of light from its flannel, and blue and blue a figure in a clump of laurelop was still a hideous and joy

## Additional Reading

- A. Vaswani et al., *[Attention is All You Need](https://arxiv.org/abs/1706.03762)*, 2017
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), YouTube Video, 2023
