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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
            img = nib.load(self.data[0][1]).dataobj[:,:,:,70]
            nib.save(nib.Nifti1Image(img, np.eye(4)), f"sample_{self.mode}.nii")

        print(f"number of {self.mode} subj: {len(subjects)}")
        print(f"length of {self.mode} samples: {len(self.data)}")
        print(f"ADNISwiFTDataset: Prepared {len(self.data)} sequences for {'training' if self.train else 'validation/testing'}.")
    
    def split_subjects(self):
        all_subjects = dict()

        meta_df = pd.read_csv(self.config['csv_path'], usecols=['ID', 'Subject', 'Sex', 'Age', 'Path_fMRI_brain'])
        
        # Filtering
        print(f"Filtering data for {self.config['downstream_task']} task...")
        meta_df = meta_df[(meta_df['Age'] < 69) | (meta_df['Age'] > 78)]
        meta_df["Age"] = meta_df["Age"].apply(lambda x: 0 if x < 69 else 1)
        meta_df["Sex"] = meta_df["Sex"].apply(lambda x: 0 if x == 'F' else 1)

        # Shuffle subjects
        all_subjects = meta_df.set_index('ID')[['Subject', 'Age', 'Sex', 'Path_fMRI_brain']].apply(list, axis=1).to_dict()
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

        # num_train_target_0 = len([id for id in train_ids if all_subjects[id][1] == 0])
        # num_train_target_1 = len([id for id in train_ids if all_subjects[id][1] == 1])
        # print(f"Number of train subjects with target 0: {num_train_target_0}")
        # print(f"Number of train subjects with target 1: {num_train_target_1}")

        # num_val_target_0 = len([id for id in val_ids if all_subjects[id][1] == 0])
        # num_val_target_1 = len([id for id in val_ids if all_subjects[id][1] == 1])
        # print(f"Number of validation subjects with target 0: {num_val_target_0}")
        # print(f"Number of validation subjects with target 1: {num_val_target_1}")

        # num_test_target_0 = len([id for id in test_ids if all_subjects[id][1] == 0])
        # num_test_target_1 = len([id for id in test_ids if all_subjects[id][1] == 1])
        # print(f"Number of test subjects with target 0: {num_test_target_0}")
        # print(f"Number of test subjects with target 1: {num_test_target_1}")

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
        for _, (subject_name, age, sex, path_fmri) in subjects.items():
            for start_frame_idx in starting_timepoints:
                data.append((subject_name, path_fmri, age, sex, start_frame_idx))

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
        subject_name, path_fmri, age, sex, start_frame_idx = self.data[index]
        sex_one_hot = torch.tensor([1.0, 0.0]) if sex == 0 else torch.tensor([0.0, 1.0])

        fmri_img = nib.load(path_fmri)
        fmri_data = fmri_img.dataobj[:, :, :, start_frame_idx : start_frame_idx + 20]
        fmri_data = self.pad_4d(fmri_data)  # Pad to 120x120x120x20
        fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
        # fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
        fmri_data = fmri_data.unsqueeze(0)  # Add channel dimension, now shape is (1, 120, 120, 120, 20)
        # print("Min max and mean of fmri_data:", fmri_data.min(), fmri_data.max(), fmri_data.mean())
        
        return fmri_data, sex_one_hot, age
    
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

        # self.output_head = mlp(num_classes=2, num_tokens = num_tokens)

    def forward(self, x):
        x = self.model(x)           # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        # x = self.output_head(x)     # ([8, 288, 2, 2, 2, 20]) -> ([8, 1])   
        return x

    def get_embeddings(self, x):
        x = self.model(x)           # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        x = x.view(x.size(0), -1)   # Flattens ([8, 288, 2, 2, 2, 20]) -> ([8, 4608])
        return x

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        self.image_encoder = Model(config).to(self.device)
        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding).to(self.device)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding).to(self.device)

        # Load weights and rename keys
        image_encoder_state_dict = torch.load(config['best_model_age_group'])
        new_image_encoder_state_dict = {
            k: v for k, v in image_encoder_state_dict.items() if k.startswith('model.')
        }
        
        # Use weights and freeze model
        self.image_encoder.eval()
        self.image_encoder.load_state_dict(new_image_encoder_state_dict)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, sources, targets):
        sources, targets = sources.to(self.device), targets.to(self.device)

        with torch.no_grad():
            sources = self.image_encoder(sources).view(sources.size(0), -1)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # L2 Normalization of embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Cosine similarity, multiplication is (batch_size, 256) @ (256, batch_size) = (batch_size, batch_size)
        logits = (text_embeddings @ image_embeddings.T) * torch.exp(self.temperature)

        # Defines the label index to be maximized on the diagonal (image 1 should match with text 1, ...)
        # Each label indicates the "correct" index in the logits row that should be maximized for each text-image pair
        # Create an array of indices from 0 to batch_size
        labels = torch.arange(logits.shape[0]).to(self.device) ## shape[0] is batch_size (64)

        # Calculate loss in both directions and average them
        # cross-entropy loss is used to maximize the similarity between matching pairs (diagonal elements of logits)
        # and minimize it for non-matching pairs (off-diagonal elements).
        texts_loss = F.cross_entropy(logits, labels)
        images_loss = F.cross_entropy(logits.T, labels)
        loss =  (images_loss + texts_loss) / 2.0

        return loss
    
class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 2048 for image and 768 for text, projection_dim is 1024
        self.projection = nn.Linear(embedding_dim, config["projection_dim"])
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)
    
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
        # self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.log_interval = len(self.dataloader) // 10  # Log every 10% of batches

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")
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

        for i, (fmri_sequence, sex, age) in enumerate(self.dataloader):
            fmri_sequence, sex = fmri_sequence.to(self.device), sex.to(self.device)  ## (batch_size, 64, 64, 48, 140) and (batch_size)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = self.model(fmri_sequence, sex)
            
            self.optimizer.zero_grad(set_to_none=True) # Modestly improve performance
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            # predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()
            # correct += (predicted_labels == target).sum().item()
            total += sex.size(0)  # returns the batch size

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
            for i, (fmri_sequence, sex, age) in enumerate(self.val_dataloader):
                fmri_sequence, sex = fmri_sequence.to(self.device), sex.to(self.device)  ## (batch_size, 64, 64, 48, 140) and (batch_size)
                loss = self.model(fmri_sequence, sex)
                val_loss += loss.item()
                
            avg_val_loss = val_loss / len(self.val_dataloader)
            print(f"[VALIDATION] epoch {epoch}\t| total_batch {i}\t| val_loss {avg_val_loss:.5f}")
            wandb.log({"epoch": epoch, "val_loss": round(avg_val_loss, 5)})

    def evaluate_classification_accuracy(self, dataloader):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0


        with torch.no_grad():
            # Pass through the text_projection and normalize, just like in forward()
            female_one_hot = torch.tensor([1.0, 0.0], device=self.device).float().unsqueeze(0) # Shape (1, 2)
            female_text_embedding = self.model.text_projection(female_one_hot)
            female_text_embedding = F.normalize(female_text_embedding, dim=-1) # Shape (1, config["projection_dim"])

            male_one_hot = torch.tensor([0.0, 1.0], device=self.device).float().unsqueeze(0) # Shape (1, 2)
            male_text_embedding = self.model.text_projection(male_one_hot)
            male_text_embedding = F.normalize(male_text_embedding, dim=-1) # Shape (1, config["projection_dim"])

        with torch.no_grad():
            for fmri_sequence, target_sex_one_hot, age_one_hot in tqdm(dataloader, desc="Evaluating CLIP Accuracy"):
                fmri_sequence, target_sex_one_hot = fmri_sequence.to(self.device), target_sex_one_hot.to(self.device)

                # Get image embedding
                image_features = self.model.image_encoder(fmri_sequence).view(fmri_sequence.size(0), -1)
                image_embeddings = self.model.image_projection(image_features)
                image_embeddings = F.normalize(image_embeddings, dim=-1)

                # Calculate similarities  
                # female_similarity = F.cosine_similarity(image_embeddings, female_text_embedding, dim=-1) * self.model.temperature
                # male_similarity = F.cosine_similarity(image_embeddings, male_text_embedding, dim=-1) * self.model.temperature
                female_similarity = (image_embeddings @ female_text_embedding.T) * self.model.temperature
                male_similarity = (image_embeddings @ male_text_embedding.T) * self.model.temperature

                # Predict class based on higher similarity 
                # Assuming female is target_sex_one_hot[:, 0] == 1 and male is target_sex_one_hot[:, 1] == 1
                # Adjust indexing based on your one-hot encoding definition
                predicted_labels = (male_similarity > female_similarity).squeeze() # 1 if male, 0 if female

                # female is [1,0] and male is [0,1], female is 0 and male is 1
                ground_truth_labels = target_sex_one_hot[:, 1] # 0 for female, 1 for male

                correct_predictions += (predicted_labels == ground_truth_labels).sum().item()
                total_samples += fmri_sequence.size(0)

        accuracy = correct_predictions / total_samples
        print(f"CLIP Classification Accuracy: {accuracy:.4f}")

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

def visualize_tsne_with_sex_age_groups(model, dataloader, config):
    model.eval() # Set model to evaluation mode
    device = config['device']

    all_embeddings = []
    all_sex_labels = []
    all_age_labels = []

    print("Extracting embeddings and labels for t-SNE...")
    with torch.no_grad():
        for fmri_sequence, target_sex_one_hot, age_one_hot in tqdm(dataloader):
            # Move data to device
            fmri_sequence = fmri_sequence.to(device)   # [batch, 1, 112, 112, 112, 20]
            target_sex_one_hot = target_sex_one_hot.to(device) # [batch, 2]

            # Get image embeddings from the CLIP model
            # Ensure this path matches how you get the final embeddings
            # before the similarity calculation in your CLIP forward pass
            image_features = model.image_encoder(fmri_sequence).view(fmri_sequence.size(0), -1)
            image_embeddings = model.image_projection(image_features)
            image_embeddings = F.normalize(image_embeddings, dim=-1) # Normalize as done in CLIP

            all_embeddings.append(image_embeddings.cpu().numpy())

            # Extract sex label (assuming [1,0] for female, [0,1] for male)
            # Adjust if your one-hot encoding is different
            sex_labels = torch.argmax(target_sex_one_hot, dim=1).cpu().numpy()
            all_sex_labels.append(sex_labels)

            age_labels = age_one_hot.cpu().numpy()
            all_age_labels.append(age_labels)
            
    all_embeddings = np.vstack(all_embeddings) # 1484, 256
    all_sex_labels = np.concatenate(all_sex_labels) # 1484
    all_age_labels = np.concatenate(all_age_labels) # 1484


    # 2. Create combined labels
    combined_labels_list = []
    for i in range(len(all_sex_labels)):
        sex = all_sex_labels[i] # 0 for Female, 1 for Male (based on argmax of [1,0] vs [0,1])
        age = all_age_labels[i] # 0 for Young (<69), 1 for Old (>78)

        if sex == 0 and age == 0:
            combined_labels_list.append("Female - Young")
        elif sex == 0 and age == 1:
            combined_labels_list.append("Female - Old")
        elif sex == 1 and age == 0:
            combined_labels_list.append("Male - Young")
        elif sex == 1 and age == 1:
            combined_labels_list.append("Male - Old")
    combined_labels = np.array(combined_labels_list)

    print(f"Total embeddings extracted: {len(all_embeddings)}")
    print("Performing t-SNE...")

    # 3. Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, verbose=1, perplexity=50, n_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    print("Plotting t-SNE results...")
    custom_palette = {
        "Female - Young": "red",
        "Female - Old": "blue",
        "Male - Young": "green",
        "Male - Old": "purple"
    }

    # 4. Plot with 4 colors using Seaborn
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=combined_labels,
        palette=custom_palette,
        alpha=0.8,         # Transparency
    )
    plt.title("t-SNE of fMRI Embeddings (Colored by Sex and Age Group)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig("tsne_pretrained_test_50.png")
    # plt.show()

if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open("data/config.yaml", "r"))

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

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
    
    model = CLIP(config).to(device)
    dataset_train = ADNISwiFTDataset(config, "train", generate_data=True)
    dataset_val = ADNISwiFTDataset(config, "val", generate_data=False)
    dataset_test = ADNISwiFTDataset(config, "test", generate_data=False)
    trainer = Trainer(config, model, dataset_train, dataset_val)
    # trainer.run()

    model.load_state_dict(torch.load(config['CLIP_model_path']))
    tsne_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
    trainer.evaluate_classification_accuracy(tsne_dataloader) # Train = 98.52, Val = 97.46, Testing = 97.87 | Train = 92.25, Val = 91.43, Testing = 
    # visualize_tsne_with_sex_age_groups(model, tsne_dataloader, config) 