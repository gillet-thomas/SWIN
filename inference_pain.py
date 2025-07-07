import os
import yaml
import torch
import datetime
import seaborn as sns
from tqdm import tqdm

import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from SWIN import Model, ADNISwiFTDataset
# Standard library imports
import os
import pickle

# Third-party imports
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset


# Pain study Marian Dataset
class PainDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.csv_path = './data/paths_dataset_pain.csv'
        self.batch_size = config['eval_batch_size']
        self.selected_groups = ['EMCI', 'CN', 'LMCI', 'AD'] # Not used on marian's dataset
        
        self.data = self.generate_data()
        self.data = self.get_timepoints(self.data)
        
        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Gender', 'Age', 'Age_Group', 'Pain_Distraction_Group'])
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")       # 178
        
        all_samples = []
        print("Processing data...")
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
            all_samples.append((subject, fmri_path, gender, age, age_group, pain_group))
        
        print(f"Processed {len(all_samples)} samples")          # 78820
        return all_samples

    def get_timepoints(self, subjects):
        data = []

        starting_timepoints = np.arange(0, 140, self.config['sequence_length'])
        for (subject, fmri_path, gender, age, age_group, pain_group) in subjects:
            for start_frame_idx in starting_timepoints:
                data.append((subject, gender, fmri_path, start_frame_idx))

        # (subj_id, current_sequence_frame_paths, sequence_label, start_frame_idx, num_frames_for_subject)
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

    def __getitem__(self, idx):
        subject, gender, fmri_path, start_frame_idx = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.dataobj[:, :, :, start_frame_idx:start_frame_idx + self.config['sequence_length']]
            fmri_tensor = self.pad_4d(fmri_data)
            fmri_tensor = (fmri_tensor - fmri_tensor.min()) / (fmri_tensor.max() - fmri_tensor.min() + 1e-8)
            # fmri_tensor = (fmri_tensor - fmri_tensor.mean()) / (fmri_tensor.std() + 1e-8)
            fmri_tensor = fmri_tensor.unsqueeze(0) # Add channel dimension

            # Encode gender
            gender_encoded = torch.tensor(0 if gender == 'F' else 1) # 0 if x == 'F' else 1

            # age = torch.tensor(age)
            # age_encoded = torch.tensor(age_group - 1)  # Convert 1, 2 to 0, 1

            return fmri_tensor, gender_encoded
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)

# Set random seed
config = yaml.safe_load(open("data/config.yaml", "r"))
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Load model
cuda_id = 2
best_model_path = config[f'best_swin_{config["map_task"]}']
model = Model(config).to(device=f"cuda:{cuda_id}")
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"Using model from {best_model_path}")

# Run t-SNE on validation, training and test datasets
dataset = PainDataset(config, mode='val')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
num_test_target_0 = len([id for id in dataset.data if id[1] == 'M'])
num_test_target_1 = len([id for id in dataset.data if id[1] == 'F'])
print(f"Number of test subjects with target 0: {num_test_target_0}")
print(f"Number of test subjects with target 1: {num_test_target_1}")

# num_test_target_2 = len([id for id in dataset.data if id[2] < 69])
# print(f"Number of test subjects with age_group: {num_test_target_2}")
# num_test_target_3 = len([id for id in dataset.data if id[2] > 78])
# print(f"Number of test subjects with age_group: {num_test_target_3}")
        
correct = 0
total = 0
for i, (fmri, gender) in enumerate(tqdm(data_loader)):
    fmri, gender = fmri.to(device=f"cuda:{cuda_id}"), gender.to(device=f"cuda:{cuda_id}")
    
    outputs = model(fmri)
    predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()
    # print("prediction and target gender:", predicted_labels.item(), gender.item())
    correct += (predicted_labels == gender).sum().item()
    total += gender.size(0)

accuracy = correct / total
print(f"Validation accuracy: {accuracy:.4f}, {correct}/{total}") # 62.95

# dataset = ADNISwiFTDataset(config, mode='test')
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
# num_test_target_0 = len([id for id in dataset.data if id[1] == 0])
# num_test_target_1 = len([id for id in dataset.data if id[1] == 1])
# print(f"Number of test subjects with target 0: {num_test_target_0}")
# print(f"Number of test subjects with target 1: {num_test_target_1}")
        
# correct = 0
# total = 0
# for i, (fmri, target) in enumerate(tqdm(data_loader)):
#     fmri, target = fmri.to(device=f"cuda:{cuda_id}"), target.to(device=f"cuda:{cuda_id}")
    
#     outputs = model(fmri)
#     predicted_labels = (torch.sigmoid(outputs) >= 0.5).long()
#     correct += (predicted_labels == target).sum().item()
#     total += target.size(0)

# accuracy = correct / total
# print(f"Validation accuracy: {accuracy:.4f}, {correct}/{total}") # 96.33
