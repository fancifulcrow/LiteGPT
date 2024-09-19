# LiteGPT

LiteGPT is a lightweight implementation of the GPT architecture, designed for both learning and experimentation. This repository provides the code necessary to train your own GPT-like models, with a focus on simplicity and adaptability.

## Requirements

Install the dependencies by running:
```
pip install torch tiktoken numpy matplotlib pyyaml tqdm
```

## Configuration

You can customize your model through the `config/config.yaml` file. It allows you to modify the following:

- **Model Architecture:** Dimension of feedforward layers, number of layers, number of attention heads, embedding size, context length, dropout
- **Training Parameters:** Learning rate, batch size, number of epochs.
- **Dataset Settings:** Path to your dataset, stride

## The Pre-Trained Model

The pre-trained LiteGPT model is available for download [here](#). It has almost **28 million** parameters.

### Tokenization
LiteGPT utilizes the **GPT-2 tokenizer** from the `tiktoken` library.

### Dataset
This model has been trained on a dataset featuring texts from the following books, sourced from [Project Gutenberg](https://www.gutenberg.org/):

- *Frankenstein* by Mary Shelley
- *The Adventures of Sherlock Holmes* by Arthur Conan Doyle
- *Metamorphosis* by Franz Kafka
- *The Great Gatsby* by F. Scott Fitzgerald

### Training Loss Curve
The model was trained for 100 epochs. Below is the training curve of the loss over training steps.

<p align="center">
<img src="images/training_loss.png" alt="training loss">
</p>

### Evalutation
The model achieve the following metrics on the test set:
- **Cross Entropy Loss:** 3.7223
- **Top-5 Accuracy:** 55.9533%
- **Perplexity**: 41.3594

### Generations

prompt: ``

```

```

## Additional Reading

- A. Vaswani et al., *[Attention is All You Need](https://arxiv.org/abs/1706.03762)*, 2017
- T. Brown et al., *[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)*, 2020
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), YouTube Video, 2023
