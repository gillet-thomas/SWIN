import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.helpers import get_timepoints, load_and_process_fmri, save_datasets, save_subjects


class ADNISwiFTDataset(Dataset):
    def __init__(self, config, mode, generate_data=False):
        super().__init__()
        self.config = config
        self.mode = mode
        self.train = True if mode == "train" else False

        if generate_data:
            subjects = self.generate_data()

        with open(f"{self.config['path_pickle_dataset']}/data_{self.mode}.pkl", "rb") as f:
            subjects = pickle.load(f)
            self.data = get_timepoints(subjects, sequence_length=self.config["sequence_length"])
            # self.data is subject, target, path_fmri, start_frame_idx

        if generate_data:
            img = nib.load(self.data[0][2]).dataobj[:, :, :, 70]
            nib.save(nib.Nifti1Image(img, np.eye(4)), f"./visualization/sample_{self.mode}_adni.nii")

        print(f"number of {self.mode} subj: {len(subjects)}")
        print(f"length of {self.mode} samples: {len(self.data)}")
        print(
            f"ADNISwiFTDataset: Prepared {len(self.data)} sequences for {'training' if self.train else 'validation/testing'}."
        )

    def generate_data(self):
        all_subjects = dict()

        meta_df = pd.read_csv(self.config["csv_path_adni"], usecols=["ID", "Subject", "Group", "Path_fMRI_brain"])

        # Filtering
        print(f"Filtering data for {self.config['downstream_task']} task...")
        meta_df = meta_df[(meta_df["Group"] == "AD") | (meta_df["Group"] == "CN")]
        # meta_df = meta_df[(meta_df['Age'] < 69) | (meta_df['Age'] > 78)]
        # meta_df["age"] = meta_df["Age"].apply(lambda x: 0 if x < 69 else 1)
        # meta_df["sex"] = meta_df["Sex"].apply(lambda x: 0 if x == 'F' else 1)

        # Shuffle subjects
        all_subjects = meta_df.set_index("ID")[["Subject", "Group", "Path_fMRI_brain"]].apply(list, axis=1).to_dict()
        subjects_list = list(all_subjects.keys())
        np.random.shuffle(subjects_list)

        # Compute number of subjects for each split
        total_unique_subjects = len(subjects_list)
        num_train_subjects = int(total_unique_subjects * self.config["train_split"])
        num_val_subjects = int(total_unique_subjects * self.config["val_split"])

        # Split unique subjects into train, validation, and test sets
        train_ids = subjects_list[:num_train_subjects]
        val_ids = subjects_list[num_train_subjects : num_train_subjects + num_val_subjects]
        test_ids = subjects_list[num_train_subjects + num_val_subjects :]

        num_train_target_0 = len([id for id in train_ids if all_subjects[id][1] == "CN"])
        num_train_target_1 = len([id for id in train_ids if all_subjects[id][1] == "AD"])
        print(f"Number of train subjects with target 0: {num_train_target_0}")
        print(f"Number of train subjects with target 1: {num_train_target_1}")
        total_samples = num_train_target_0 + num_train_target_1
        weight_0 = total_samples / (2 * num_train_target_0)
        weight_1 = total_samples / (2 * num_train_target_1)
        self.training_class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(self.config["device"])

        num_val_target_0 = len([id for id in val_ids if all_subjects[id][1] == "CN"])
        num_val_target_1 = len([id for id in val_ids if all_subjects[id][1] == "AD"])
        print(f"Number of validation subjects with target 0: {num_val_target_0}")
        print(f"Number of validation subjects with target 1: {num_val_target_1}")

        num_test_target_0 = len([id for id in test_ids if all_subjects[id][1] == "CN"])
        num_test_target_1 = len([id for id in test_ids if all_subjects[id][1] == "AD"])
        print(f"Number of test subjects with target 0: {num_test_target_0}")
        print(f"Number of test subjects with target 1: {num_test_target_1}")

        # Save datasets and subjects
        save_datasets(self.config["path_pickle_dataset"], all_subjects, train_ids, val_ids, test_ids)
        save_subjects(self.config["path_pickle_dataset"], train_ids, val_ids, test_ids)
        print("Datasets saved!")

    def __getitem__(self, index):
        # Unpack the data tuple for one sequence
        subject_name, group, path_fmri, start_frame_idx = self.data[index]

        target = torch.tensor(0 if group == "AD" else 1)
        fmri_data = load_and_process_fmri(path_fmri, start_frame_idx, self.config["img_size"])

        return fmri_data, target

    def __len__(self):
        return len(self.data)
