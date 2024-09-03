import os
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F

inp_means = torch.tensor([6.2446e-08, 1.4142e+01, 1.4631e+00])
inp_stds = torch.tensor([4.9318e-08, 7.2629e+00, 1.1194e+00])

target_mean = torch.tensor([0.0011250363299623132])
target_std = torch.tensor([4.9318e-08, 7.2629e+00, 1.1194e+00])

class NormalizeTransform:
    def __init__(self, input_means, input_stds, target_mean, target_std):
        self.input_means = input_means
        self.input_stds = input_stds
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, input_data, target):
        normalized_input = (input_data - self.input_means[:, None, None]) / self.input_stds[:, None, None]
        normalized_target = (target - self.target_mean) / self.target_std
        return normalized_input, normalized_target


class CustomDataset(Dataset):
    def __init__(self, root_path):
        self.data_files = self._find_files(root_path)
        self.target_size = 256
        self.norm_transform = NormalizeTransform(inp_means,inp_stds,target_mean,target_std)
    
    def _find_files(self, root_path):
        data_files = []

        current_files = glob.glob(os.path.join(root_path, '*_current.csv.gz'))
        eff_dist_files = glob.glob(os.path.join(root_path, '*_eff_dist.csv.gz'))
        pdn_density_files = glob.glob(os.path.join(root_path, '*_pdn_density.csv.gz'))
        ir_drop_files = glob.glob(os.path.join(root_path, '*_ir_drop.csv.gz'))

        current_files.sort()
        eff_dist_files.sort()
        pdn_density_files.sort()
        ir_drop_files.sort()

        for current, eff_dist, pdn_density, ir_drop in zip(current_files, eff_dist_files, pdn_density_files, ir_drop_files):
            data_files.append({
                'current': current,
                'eff_dist': eff_dist,
                'pdn_density': pdn_density,
                'ir_drop': ir_drop
            })

        return data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_group = self.data_files[idx]

        current = pd.read_csv(file_group['current'], compression='gzip').to_numpy()
        eff_dist = pd.read_csv(file_group['eff_dist'], compression='gzip').to_numpy()
        pdn_density = pd.read_csv(file_group['pdn_density'], compression='gzip').to_numpy()
        ir_drop = pd.read_csv(file_group['ir_drop'], compression='gzip').to_numpy()

                # Convert to torch tensors and add batch and channel dimensions
        current = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        eff_dist = torch.tensor(eff_dist, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pdn_density = torch.tensor(pdn_density, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ir_drop = torch.tensor(ir_drop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Resize tensors
        current = F.interpolate(current, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        eff_dist = F.interpolate(eff_dist, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        pdn_density = F.interpolate(pdn_density, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        ir_drop = F.interpolate(ir_drop, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            
        current = current/current.max()
        eff_dist = eff_dist/eff_dist.max()
        pdn_density = pdn_density/pdn_density.max()
        ir_drop = ir_drop/ir_drop.max()
            
        # input_data = {
        #     'current':current ,
        #     'eff_dist': eff_dist,
        #     'pdn_density': pdn_density
        # }
        input_data = torch.stack([current, eff_dist, pdn_density], dim=0)
        input_data=input_data[:,0,:,:] # 3,256,256 
        target = ir_drop
        # input_data,target = self.norm_transform(input_data,target)

        return input_data, target


def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_state=42):
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), " 1.0"
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_valid_size = int((train_ratio + valid_ratio) * dataset_size)
    train_valid_indices, test_indices = train_test_split(indices, train_size=train_valid_size, random_state=random_state)
    
    train_size = int(train_ratio * dataset_size)
    train_indices, valid_indices = train_test_split(train_valid_indices, train_size=train_size, random_state=random_state)
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, valid_dataset, test_dataset

def build_dataset(root_path=r'F:\NowWorking\Job_IR\data\BeGAN-circuit-benchmarks\nangate45\set1\data'):
    dataset = CustomDataset(root_path)
    return split_dataset(dataset)
