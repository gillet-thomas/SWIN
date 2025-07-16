import datetime
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb


class Trainer:
    def __init__(self, config, model, dataset_train, dataset_val):
        self.config = config
        self.device = config["device"]
        self.model = model.to(self.device)
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.data = dataset_train
        self.val_data = dataset_val
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        self.scaler = torch.amp.GradScaler()  # for Automatic Mixed Precision
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.log_interval = len(self.dataloader) // 10  # Log every 10% of batches

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)"
        )
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")
        print("=" * 50)

    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./results/runs/{timestamp}"
        os.mkdir(path) if not os.path.exists(path) else None

        print(f"Running on device: {self.device}")
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)
            torch.save(self.model.state_dict(), f"{path}/model-e{epoch}.pth")
            torch.save(self.model.state_dict(), f"./results/runs/last_model.pth")
            print(f"MODEL SAVED to .{path}/model-e{epoch}.pth")

    def train(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (fmri_sequence, target) in enumerate(self.dataloader):
            fmri_sequence, target = fmri_sequence.to(self.device), target.to(self.device)
            fmri_sequence, target = fmri_sequence.float(), target.float()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(fmri_sequence)
                outputs = outputs.view(-1)  # for BCEWithLogitsLoss
                loss = self.criterion(outputs, target)

            self.optimizer.zero_grad(set_to_none=True)  # Modestly improve performance
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()  # BCEWithLogitsLoss
            # predicted_labels = torch.argmax(outputs, dim=1) # CrossEntropyLoss
            correct += (predicted_labels == target).sum().item()
            total += target.size(0)  # returns the batch size

            if i != 0 and i % self.log_interval == 0:
                avg_loss = round(running_loss / self.log_interval, 5)
                accuracy = round(correct / total, 5)
                lr = round(self.optimizer.param_groups[0]["lr"], 5)
                print(
                    f"epoch {epoch}\t| batch {i}/{len(self.dataloader)}\t| train_loss: {avg_loss}\t| train_accuracy: {accuracy}\t| learning_rate: {lr}"
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": i,
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "learning_rate": lr,
                    }
                )
                correct, total, running_loss = 0, 0, 0.0

    def validate(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for i, (fmri_sequence, target) in enumerate(self.val_dataloader):
                fmri_sequence, target = fmri_sequence.to(self.device), target.to(self.device)
                fmri_sequence, target = fmri_sequence.float(), target.float()

                outputs = self.model(fmri_sequence)
                outputs = outputs.view(-1)  # for BCEWithLogitsLoss
                loss = self.criterion(outputs, target)
                val_loss += loss.item()

                predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()
                correct += (predicted_labels == target).sum().item()
                total += target.size(0)  # returns the batch size

            avg_val_loss = val_loss / len(self.val_dataloader)
            accuracy = correct / total
            print(
                f"[VALIDATION] epoch {epoch}\t| total_batch {i}\t| val_loss {avg_val_loss:.5f}\t| val_accuracy {accuracy:.5f}"
            )
            wandb.log({"epoch": epoch, "val_loss": round(avg_val_loss, 5), "val_accuracy": round(accuracy, 5)})
