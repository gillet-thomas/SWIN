import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from itertools import product
import nibabel as nib

from torchvision import transforms
from nibabel import load, Nifti1Image, save
from torch.nn import functional as F
    



class ADNISwiFTDataset(Dataset): # Inherit from BaseDataset
    def __init__(self, config, mode):
        super().__init__()   
        self.config = config 
        self.mode = mode
        self.train = True if mode == 'train' else False

        subjects = self.get_subjects()
        self.save_split(subjects)
        self.data = self.get_timepoints(subjects) # subject, target, path_fmri, start_frame_idx

        print(f"number of {self.mode} subj: {len(subjects)}")
        print(f"length of {self.mode} samples: {len(self.data)}")
        print(f"ADNISwiFTDataset: Prepared {len(self.data)} sequences for {'training' if self.train else 'validation/testing'}.")
    
    def save_split(self, subjects):
        with open(f"data/{self.mode}.txt", "w") as f:
            for id in subjects.keys():
                f.write(f"{id}\n")
    
    def get_subjects(self):
        all_subjects = dict()
        meta_df = pd.read_csv(self.config['image_path'], usecols=['ID', 'Subject', 'Sex', 'Age', 'Path_fMRI_brain'])
        # meta_df = pd.read_csv(self.config['image_path'], usecols=['ID', 'subject_id', 'Sex', 'Age', 'Path_fMRI_brain'])
        
        # Filtering
        if self.config['downstream_task'] == 'age_group':
            meta_df = meta_df[(meta_df['Age'] < 69) | (meta_df['Age'] > 78)]
            meta_df["target"] = meta_df["Age"].apply(lambda x: 0 if x < 69 else 1)
        elif self.config['downstream_task'] == 'sex':
            meta_df["target"] = meta_df["Sex"].apply(lambda x: 0 if x == 'F' else 1)

        # Shuffle subjects
        all_subjects = meta_df.set_index('ID')[['Subject', 'target', 'Path_fMRI_brain']].apply(list, axis=1).to_dict()
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

        splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
        return {id: all_subjects[id] for id in splits[self.mode]}

    def get_timepoints(self, subjects):
        data = [] # This will store the final list of (subj_id, current_sequence_frame_paths, sequence_label, start_frame_idx, num_frames_for_subject) tuples

        starting_timepoints = np.arange(0, 140, 20)
        for _, (subject_name, target, path_fmri) in subjects.items():
            for start_frame_idx in starting_timepoints:
                data.append((subject_name, target, path_fmri, start_frame_idx))

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
        subject_name, target, path_fmri, start_frame_idx = self.data[index]

        fmri_img = nib.load(path_fmri)
        fmri_data = fmri_img.dataobj[:, :, :, start_frame_idx : start_frame_idx + 20]
        # fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
        fmri_data = self.pad_4d(fmri_data)  # Pad to 120x120x120x20
        fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
        fmri_data = fmri_data.unsqueeze(0)  # Add channel dimension, now shape is (1, 120, 120, 120, 20)
        # print("Min max and mean of fmri_data:", fmri_data.min(), fmri_data.max(), fmri_data.mean())
        target = torch.tensor(target).float() # For classification, labels should be LongTensor
        return fmri_data, target
    
    def __len__(self):
        return len(self.data)



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, img_norm):
        self.img_norm = img_norm
        self.data_df = data_df
        all_combinations = product(self.data_df['Subject_ImageID'].unique(), np.arange(0, 140, 20))
        combinations_df = pd.DataFrame(all_combinations, columns=['Subject_ImageID', 'initial_block'])
        
        df_merged = pd.merge(combinations_df, self.data_df[['Subject_ImageID', 'Image_path', 'Target']], on='Subject_ImageID', how='left')
        self.data_df = df_merged
        values = data_df['Target'].value_counts()
        print(values)
        print(100 * values[0] / (values[0] + values[1]), 100 - 100 * values[0] / (values[0] + values[1]))

    def _resize(self, img):
        assert img.shape == (91, 109, 91, 20), f"Expected shape (91,109,91,20), got {img.shape}"
        img = F.pad(img, pad=(0,0, 2,3, 0,0, 2,3))  # (96, 109, 96, 20)
        img = img[:, 6:6+96, :, :]  # (96, 96, 96, 20)
        return img

    def _normalize_4d(self, img, img_norm):
        if img_norm == 'minmax':
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        elif img_norm == 'znorm':
            img = (img - img.mean()) / (img.std() + 1e-8)
        img = img.unsqueeze(0)  # (1, 1, D, H, W, T)
        return img

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_path = row['Image_path']
        initial_block = row['initial_block']
        img = load(img_path, mmap = True)
        img = img.dataobj[:, :, :, initial_block : initial_block + 20]
        img = torch.from_numpy(img).float()

        img = self._resize(img)
        img = self._normalize_4d(img, self.img_norm)
        
        label = self.data_df.iloc[idx]['Target']
        return img, label

    def __len__(self):
        return len(self.data_df)