import pickle

import nibabel as nib
import numpy as np
import torch


def pad_4d(fmri_data, img_size):
    background_value = fmri_data[0, 0, 0]  # Find background value
    padded_volume = np.full(img_size, background_value, dtype=fmri_data.dtype)

    pad_x = (img_size[0] - fmri_data.shape[0]) // 2
    pad_y = (img_size[1] - fmri_data.shape[1]) // 2
    pad_z = (img_size[2] - fmri_data.shape[2]) // 2

    padded_volume[
        pad_x : pad_x + fmri_data.shape[0],
        pad_y : pad_y + fmri_data.shape[1],
        pad_z : pad_z + fmri_data.shape[2],
    ] = fmri_data

    return torch.tensor(padded_volume, dtype=torch.float32)


def get_timepoints_adni(subjects, limit=140, sequence_length=20):
    data = []

    starting_timepoints = np.arange(0, limit, sequence_length)
    for _, (subject_name, group, path_fmri) in subjects.items():
        for start_frame_idx in starting_timepoints:
            data.append((subject_name, group, path_fmri, start_frame_idx))  # add start_frame_idx column

    return data


def get_timepoints_pain(subjects, limit=140, sequence_length=20):
    data = []

    starting_timepoints = np.arange(0, limit, sequence_length)
    for subject_name, (group, path_fmri) in subjects.items():
        for start_frame_idx in starting_timepoints:
            data.append((subject_name, group, path_fmri, start_frame_idx))  # add start_frame_idx column

    return data


def get_timepoints_clip(subjects, limit=140, sequence_length=20):
    data = []

    starting_timepoints = np.arange(0, limit, sequence_length)
    for _, (subject_name, age, sex, path_fmri) in subjects.items():
        for start_frame_idx in starting_timepoints:
            data.append((subject_name, path_fmri, age, sex, start_frame_idx))

    return data


def load_and_process_fmri(path_fmri, start_frame_idx, img_size):
    fmri_img = nib.load(path_fmri)
    fmri_data = fmri_img.dataobj[:, :, :, start_frame_idx : start_frame_idx + 20]
    fmri_data = pad_4d(fmri_data, img_size)  # Pad to 120x120x120x20
    fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
    # fmri_data = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
    fmri_data = fmri_data.unsqueeze(0)  # Add channel dimension, now shape is (1, 120, 120, 120, 20)
    return fmri_data


def save_datasets(path_pickle_dataset, all_subjects, train_ids, val_ids, test_ids=None):
    with open(f"{path_pickle_dataset}/data_train.pkl", "wb") as f:
        subjects = {id: all_subjects[id] for id in train_ids}
        pickle.dump(subjects, f)
    with open(f"{path_pickle_dataset}/data_val.pkl", "wb") as f:
        subjects = {id: all_subjects[id] for id in val_ids}
        pickle.dump(subjects, f)
    if test_ids is not None:
        with open(f"{path_pickle_dataset}/data_test.pkl", "wb") as f:
            subjects = {id: all_subjects[id] for id in test_ids}
            pickle.dump(subjects, f)


def save_subjects(path_pickle_dataset, train_ids, val_ids, test_ids=None):
    with open(f"{path_pickle_dataset}/train.txt", "w") as f:
        for id in train_ids:
            f.write(f"{id}\n")
    with open(f"{path_pickle_dataset}/val.txt", "w") as f:
        for id in val_ids:
            f.write(f"{id}\n")
    if test_ids is not None:
        with open(f"{path_pickle_dataset}/test.txt", "w") as f:
            for id in test_ids:
                f.write(f"{id}\n")
