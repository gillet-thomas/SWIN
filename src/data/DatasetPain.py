import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.helpers import get_timepoints, load_and_process_fmri, save_datasets


# Pain study Marian Dataset
class PainDataset(Dataset):
    def __init__(self, config, mode="train", generate_data=False):
        self.mode = mode
        self.config = config
        self.csv_path = config["csv_path_pain"]
        self.batch_size = config["batch_size"]
        self.split_ratio = config["train_split"]
        self.dataset_path = f"./src/data/data_{mode}.pkl"

        if generate_data:
            self.generate_data()

        with open(f"{self.config['path_pickle_dataset']}/data_{self.mode}.pkl", "rb") as f:
            subjects = pickle.load(f)  # 78820 train samples and 8680 val sample
            self.data = get_timepoints(subjects, sequence_length=self.config["sequence_length"])
            # self.data is subject, target, path_fmri, start_frame_idx

        if generate_data:
            img = nib.load(self.data[0][2]).dataobj[:, :, :, 70]
            nib.save(nib.Nifti1Image(img, np.eye(4)), f"./results/visualization/sample_{self.mode}_pain.nii")

        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self):
        # Load CSV file
        meta_df = pd.read_csv(self.csv_path, usecols=["Subject", "Path_fMRI", "Gender"])

        # Filtering
        print(f"Filtering data for {self.config['downstream_task']} task...")
        # meta_df = meta_df[(meta_df['Age'] <= 26) | (meta_df['Age'] >= 68)]
        # meta_df["age"] = meta_df["Age"].apply(lambda x: 0 if x < 26 else 1)
        meta_df["Gender"] = meta_df["Gender"].apply(lambda x: 0 if x == "F" else 1)

        # Shuffle subjects
        all_subjects = meta_df.set_index("Subject")[["Gender", "Path_fMRI"]].apply(list, axis=1).to_dict()
        subjects_list = list(all_subjects.keys())
        np.random.shuffle(subjects_list)

        # Compute number of subjects for each split
        total_unique_subjects = len(subjects_list)
        num_train_subjects = int(total_unique_subjects * self.config["train_split"])

        # Split unique subjects into train, validation, and test sets
        train_ids = subjects_list[:num_train_subjects]
        val_ids = subjects_list[num_train_subjects:]
        print(f"Training subjects: {len(train_ids)}")  # 24
        print(f"Validation subjects: {len(val_ids)}")  # 11

        num_train_target_0 = len([id for id in train_ids if all_subjects[id][0] == 0])
        num_train_target_1 = len([id for id in train_ids if all_subjects[id][0] == 1])
        print(f"Number of train subjects with target 0: {num_train_target_0}")
        print(f"Number of train subjects with target 1: {num_train_target_1}")

        num_val_target_0 = len([id for id in val_ids if all_subjects[id][0] == 0])
        num_val_target_1 = len([id for id in val_ids if all_subjects[id][0] == 1])
        print(f"Number of validation subjects with target 0: {num_val_target_0}")
        print(f"Number of validation subjects with target 1: {num_val_target_1}")

        # Save to pickle files
        save_datasets(self.config["path_pickle_dataset"], all_subjects, train_ids, val_ids)
        print("Datasets saved!")

    def __getitem__(self, idx):
        subject, group, path_fmri, start_frame_idx = self.data[idx]  # Types are str, torch.Tensor, str, str, int

        target = torch.tensor(group)
        fmri_data = load_and_process_fmri(path_fmri, start_frame_idx, self.config["img_size"])

        return fmri_data, target

    def __len__(self):
        return len(self.data)
