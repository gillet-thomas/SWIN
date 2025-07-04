# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer



class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        print("Root, subject_dict:", root, subject_dict)

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
                y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 
        
class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                rand_y = torch.nn.functional.pad(rand_y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                    } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()

        if self.contrastive:
            rand_y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subj,
                "target": target,
                "TR": 0,
                }
        else:
            return {
                    "fmri_sequence": y,
                    "subject_name": subj,
                    "target": target,
                    "TR": 0,
                    "sex": sex,
                    } 

