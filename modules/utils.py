import matplotlib.pyplot as plt
import yaml


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_curve(losses:list[float], title:str) -> None:
    plt.figure(figsize=(16, 9))
    plt.plot(losses)
    plt.title(f"{title.title()}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.savefig("loss_curve.png")


def load_configuration(config_path:str) -> dict:
    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)

    return config
