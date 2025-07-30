import argparse
import os
import warnings
from typing import Annotated, Tuple

import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from src.data.DatasetADNI import ADNISwiFTDataset
from src.data.DatasetPain import PainDataset
from src.models.SWIN4D import SWIN4D
from src.Trainer import Trainer


def parse_args() -> Annotated[argparse.Namespace, "args"]:
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


def set_seeds(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])


def update_config(config: dict, args: argparse.Namespace) -> Annotated[dict, "config"]:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    config["wandb_mode"] = "online" if args.wandb else "disabled"
    config["downstream_task"] = args.task  # Update config with task from args
    return config


def init_trackers(config: dict, args: argparse.Namespace, DDP=False):
    wandb.init(
        project="fMRI2Vec",
        config=config,
        name=args.name,
        mode="online" if config["wandb_mode"] == "online" else "disabled",
        group="DDP" if DDP else None,
    )

    # mlflow.set_tracking_uri(uri="http://localhost:8080")
    # mlflow.set_experiment(experiment_name="fMRI2Vec")
    # mlflow.start_run(run_name=args.name)
    # mlflow.log_params(config)


def load_data(
    config: dict,
) -> Tuple[Annotated[ADNISwiFTDataset, "train_dataset"], Annotated[ADNISwiFTDataset, "val_dataset"]]:
    train_dataset = ADNISwiFTDataset(config, "train", generate_data=config["generate_data"])
    val_dataset = ADNISwiFTDataset(config, "val", generate_data=False)
    return train_dataset, val_dataset


def init_model(config: dict) -> Annotated[SWIN4D, "SWIN4D model"]:
    model = SWIN4D(config).to(config["device"])
    return model


def train_model(config: dict, model, train_dataset, val_dataset):
    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.run()


####! Single GPU training
def SWIN_pipeline():
    args = parse_args()
    config = yaml.safe_load(open("configs/config.yaml", "r"))

    set_seeds(config)
    config = update_config(config, args)
    init_trackers(config, args)

    train_dataset, val_dataset = load_data(config)
    model = init_model(config)
    train_model(config, model, train_dataset, val_dataset)


####! DDP training
def DDP_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def DDP_cleanup():
    dist.destroy_process_group()


def DDP_SWIN_pipeline(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    DDP_setup(rank, world_size)

    args = parse_args()
    config = yaml.safe_load(open("configs/config.yaml", "r"))

    set_seeds(config)
    config = update_config(config, args)
    config["device"] = rank
    init_trackers(config, args, DDP=True)

    train_dataset, val_dataset = load_data(config)
    model = init_model(config)
    DDP_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    train_model(config, DDP_model, train_dataset, val_dataset)

    DDP_cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def DDP_run(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Run on single GPU
    SWIN_pipeline()

    # Run on multiple GPUs (DDP)
    # n_gpus = torch.cuda.device_count()
    # DDP_run(DDP_SWIN_pipeline, n_gpus)
