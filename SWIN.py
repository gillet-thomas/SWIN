import argparse

import numpy as np
import torch
import yaml

import wandb
from src.data.DatasetADNI import ADNISwiFTDataset
from src.data.DatasetPain import PainDataset
from src.models.SWIN4D import SWIN4D
from src.Trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate fMRI Model")
    parser.add_argument("name", type=str, nargs="?", default=None, help="WandB run name (optional)")
    parser.add_argument("--task", type=str, default="age_group", help="Task to run (age_group or sex)")
    parser.add_argument("--cuda", type=int, default=2, help="CUDA device to use (e.g., 0 for GPU 0)")
    parser.add_argument(
        "--wandb",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable Weights and Biases (WandB) tracking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open("configs/config.yaml", "r"))

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    config["device"] = args.cuda
    config["wandb_mode"] = "online" if args.wandb else "disabled"
    config["downstream_task"] = args.task  # Update config with task from args
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="fMRI2Vec",
        config=config,
        name=args.name,
        mode="online" if config["wandb_mode"] == "online" else "disabled",
    )

    model = SWIN4D(config).to(device)
    dataset_train = ADNISwiFTDataset(config, "train", generate_data=True)
    dataset_val = ADNISwiFTDataset(config, "val", generate_data=False)

    trainer = Trainer(config, model, dataset_train, dataset_val)
    trainer.run()
