import os
import yaml
import torch
import wandb
import pickle
import datetime
import torch.nn as nn
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from torch.nn import functional as F
from torch.utils.data import Dataset

from project.module.models.clf_mlp import mlp
from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D


class ADNISwiFTDataset(Dataset):
    def __init__(self, config, mode, generate_data=False):
        super().__init__()   
        self.config = config 
        self.mode = mode
        self.train = True if mode == 'train' else False

        if generate_data:
            subjects = self.split_subjects()

        with open(f"data/data_{self.mode}.pkl", "rb") as f:
            subjects = pickle.load(f)
            self.data = self.get_timepoints(subjects) # subject, target, path_fmri, start_frame_idx
        
        if generate_data:
            img = nib.load(self.data[0][2]).dataobj[:,:,:,70]
            nib.save(nib.Nifti1Image(img, np.eye(4)), f"sample_{self.mode}.nii")

        print(f"number of {self.mode} subj: {len(subjects)}")
        print(f"length of {self.mode} samples: {len(self.data)}")
        print(f"ADNISwiFTDataset: Prepared {len(self.data)} sequences for {'training' if self.train else 'validation/testing'}.")
    
    def split_subjects(self):
        all_subjects = dict()

        meta_df = pd.read_csv(self.config['csv_path'], usecols=['ID', 'Subject', 'Group', 'Path_fMRI_brain'])
        
        # Filtering
        print(f"Filtering data for {self.config['downstream_task']} task...")
        meta_df = meta_df[(meta_df['Group'] == 'AD') | (meta_df['Group'] == 'CN')]

        # Shuffle subjects
        all_subjects = meta_df.set_index('ID')[['Subject', 'Group', 'Path_fMRI_brain']].apply(list, axis=1).to_dict()
        subjects_list = list(all_subjects.keys())
        np.random.shuffle(subjects_list)

        # Compute number of subjects for each split
        total_unique_subjects = len(subjects_list)
        num_train_subjects = int(total_unique_subjects * self.config['train_split'])
        num_val_subjects = int(total_unique_subjects * self.config['val_split'])

        # Split unique subjects into train, validation, and test sets
        train_ids = subjects_list[:num_train_subjects]
        val_ids = subjects_list[num_train_subjects : num_train_subjects + num_val_subjects]
        test_ids = subjects_list[num_train_subjects + num_val_subjects:]

        # Save to pickle files
        with open("./data/data_train.pkl", 'wb') as f:
            subjects = {id: all_subjects[id] for id in train_ids}
            pickle.dump(subjects, f)
        with open("./data/data_val.pkl", 'wb') as f:
            subjects = {id: all_subjects[id] for id in val_ids}
            pickle.dump(subjects, f)
        with open("./data/data_test.pkl", 'wb') as f:
            subjects = {id: all_subjects[id] for id in test_ids}
            pickle.dump(subjects, f)

        num_train_target_0 = len([id for id in train_ids if all_subjects[id][1] == 'CN'])
        num_train_target_1 = len([id for id in train_ids if all_subjects[id][1] == 'AD'])
        print(f"Number of train subjects with target 0: {num_train_target_0}")
        print(f"Number of train subjects with target 1: {num_train_target_1}")
        total_samples = num_train_target_0 + num_train_target_1
        weight_0 = total_samples / (2 * num_train_target_0)
        weight_1 = total_samples / (2 * num_train_target_1)
        self.training_class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(self.config['device'])

        num_val_target_0 = len([id for id in val_ids if all_subjects[id][1] == 'CN'])
        num_val_target_1 = len([id for id in val_ids if all_subjects[id][1] == 'AD'])
        print(f"Number of validation subjects with target 0: {num_val_target_0}")
        print(f"Number of validation subjects with target 1: {num_val_target_1}")

        num_test_target_0 = len([id for id in test_ids if all_subjects[id][1] == 'CN'])
        num_test_target_1 = len([id for id in test_ids if all_subjects[id][1] == 'AD'])
        print(f"Number of test subjects with target 0: {num_test_target_0}")
        print(f"Number of test subjects with target 1: {num_test_target_1}")

        # Save subjects to txt files
        with open("data/train.txt", "w") as f:
            for id in train_ids:
                f.write(f"{id}\n")
        with open("data/val.txt", "w") as f:
            for id in val_ids:
                f.write(f"{id}\n")
        with open("data/test.txt", "w") as f:
            for id in test_ids:
                f.write(f"{id}\n")

        print("Datasets saved!")
                
    def get_timepoints(self, subjects):
        data = []

        starting_timepoints = np.arange(0, 140, self.config['sequence_length'])
        for _, (subject_name, group, path_fmri) in subjects.items():
            for start_frame_idx in starting_timepoints:
                data.append((subject_name, group, path_fmri, start_frame_idx)) # add start_frame_idx column

        return data

    def pad_4d(self, fmri_data):
        background_value = fmri_data[0,0,0] # Find background value
        padded_volume = np.full(self.config['img_size'], background_value, dtype=fmri_data.dtype)

        pad_x = (self.config['img_size'][0] - fmri_data.shape[0]) // 2
        pad_y = (self.config['img_size'][1] - fmri_data.shape[1]) // 2
        pad_z = (self.config['img_size'][2] - fmri_data.shape[2]) // 2

        padded_volume[pad_x : pad_x + fmri_data.shape[0],
                        pad_y : pad_y + fmri_data.shape[1],
                        pad_z : pad_z + fmri_data.shape[2]] = fmri_data
    
        return torch.tensor(padded_volume, dtype=torch.float32)
    
    def __getitem__(self, index):
        # Unpack the data tuple for one sequence
        subject_name, group, path_fmri, start_frame_idx = self.data[index]

        target = 0 if group == 'AD' else 1

        fmri_img = nib.load(path_fmri)
        fmri_data = fmri_img.dataobj[:, :, :, start_frame_idx : start_frame_idx + 20]
        fmri_data = self.pad_4d(fmri_data)  # Pad to 120x120x120x20
        fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
        # fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
        fmri_data = fmri_data.unsqueeze(0)  # Add channel dimension, now shape is (1, 120, 120, 120, 20)
        # print("Min max and mean of fmri_data:", fmri_data.min(), fmri_data.max(), fmri_data.mean())
        
        return fmri_data, torch.tensor(target)
    
    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model = SwinTransformer4D(
            img_size=config['img_size'],
            in_chans=config['in_chans'],
            embed_dim=config['embed_dim'],
            window_size=config['window_size'],
            first_window_size=config['first_window_size'],
            patch_size=config['patch_size'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            c_multiplier=config['c_multiplier'],
            last_layer_full_MSA=config['last_layer_full_MSA'],
            attn_drop_rate=config['dropout']
        )
        num_tokens = config['embed_dim'] * (config['c_multiplier'] ** (config['n_stages'] - 1))
        self.output_head = mlp(num_classes=config['num_classes'], num_tokens = num_tokens)

    def forward(self, x):
        x = self.model(x)           # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        x = self.output_head(x)     # ([8, 288, 2, 2, 2, 20]) -> ([8, 1])  
        return x

    def get_embeddings(self, x):
        x = self.model(x)           # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        x = x.view(x.size(0), -1)   # Flattens ([8, 288, 2, 2, 2, 20]) -> ([8, 4608])
        return x

class Trainer():
    def __init__(self, config, model, dataset_train, dataset_val):
        self.config = config
        self.device = config['device']
        self.model = model.to(self.device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

        self.data = dataset_train
        self.val_data = dataset_val
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

        self.scaler = torch.amp.GradScaler()       # for Automatic Mixed Precision
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.log_interval = len(self.dataloader) // 10  # Log every 10% of batches

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")          ## 114 batches of size 64
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")    ## 13 batches of size 64
        print("=" * 50)

    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./results/{timestamp}"
        os.mkdir(path) if not os.path.exists(path) else None

        print(f"Running on device: {self.device}")
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)
            torch.save(self.model.state_dict(), f'{path}/model-e{epoch}.pth')
            torch.save(self.model.state_dict(), f'./results/last_model.pth')
            print(f"MODEL SAVED to .{path}/model-e{epoch}.pth")
    
    def train(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (fmri_sequence, target) in enumerate(self.dataloader):
            fmri_sequence, target = fmri_sequence.to(self.device), target.to(self.device)
            fmri_sequence, target = fmri_sequence.float(), target.float() 

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(fmri_sequence)
                outputs = outputs.view(-1) # for BCEWithLogitsLoss
                loss = self.criterion(outputs, target)
            
            self.optimizer.zero_grad(set_to_none=True) # Modestly improve performance
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            predicted_labels = (torch.sigmoid(outputs) >= 0.5).long() # BCEWithLogitsLoss
            # predicted_labels = torch.argmax(outputs, dim=1) # CrossEntropyLoss
            correct += (predicted_labels == target).sum().item()
            total += target.size(0)  # returns the batch size

            if i != 0 and i % self.log_interval == 0:
                avg_loss = round(running_loss / self.log_interval, 5)
                accuracy = round(correct / total, 5)
                lr = round(self.optimizer.param_groups[0]['lr'], 5)
                print(f"epoch {epoch}\t| batch {i}/{len(self.dataloader)}\t| train_loss: {avg_loss}\t| train_accuracy: {accuracy}\t| learning_rate: {lr}")
                wandb.log({"epoch": epoch, "batch": i, "train_loss": avg_loss, "train_accuracy": accuracy, "learning_rate": lr})
                correct, total, running_loss = 0, 0, 0.0

    def validate(self, epoch):  
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for i, (fmri_sequence, target) in enumerate(self.val_dataloader):
                fmri_sequence, target = fmri_sequence.to(self.device), target.to(self.device) 
                fmri_sequence, target = fmri_sequence.float(), target.float()  
                
                outputs = self.model(fmri_sequence)
                outputs = outputs.view(-1) # for BCEWithLogitsLoss
                loss = self.criterion(outputs, target)
                val_loss += loss.item()
                
                predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()
                correct += (predicted_labels == target).sum().item()
                total += target.size(0)  # returns the batch size
                
            avg_val_loss = val_loss / len(self.val_dataloader)
            accuracy = correct / total
            print(f"[VALIDATION] epoch {epoch}\t| total_batch {i}\t| val_loss {avg_val_loss:.5f}\t| val_accuracy {accuracy:.5f}")
            wandb.log({"epoch": epoch, "val_loss": round(avg_val_loss, 5), "val_accuracy": round(accuracy, 5)})

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate fMRI Model")
    parser.add_argument(
        "name", type=str, nargs="?", default=None, help="WandB run name (optional)"
    )
    parser.add_argument(
        "--task", type=str, default="age_group", help="Task to run (age_group or sex)"
    )
    parser.add_argument(
        "--cuda", type=int, default=2, help="CUDA device to use (e.g., 0 for GPU 0)"
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable Weights and Biases (WandB) tracking",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open("data/config.yaml", "r"))

    config['device'] = args.cuda
    config["wandb_mode"] = "online" if args.wandb else "disabled"
    config['downstream_task'] = args.task # Update config with task from args
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="fMRI2Vec",
        config=config,
        name=args.name,
        mode="online" if config["wandb_mode"] == "online" else "disabled",
    )
    
    model = Model(config).to(device)
    dataset_train = ADNISwiFTDataset(config, "train", generate_data=True)
    dataset_val = ADNISwiFTDataset(config, "val", generate_data=False)

    trainer = Trainer(config, model, dataset_train, dataset_val)
    trainer.run()