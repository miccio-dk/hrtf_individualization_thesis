import torch
from torch.utils.data import Dataset
from .sofa_dataset import SofaDataset
from .anthro_dataset import AnthropometricsDataset

class AnthroSofaDataset(Dataset):
    def __init__(self, data_path, dataset_type, **kwargs):
        self.sofa_dataset = SofaDataset(data_path, **kwargs)
        self.anthro_dataset = AnthropometricsDataset(data_path, dataset_type)

    def __len__(self):
        return len(self.sofa_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        hrir, labels = self.sofa_dataset[idx]
        anthro_idx = (labels['ear'], labels['subj'])
        features = self.anthro_dataset[anthro_idx]
        labels.update(features)
        return hrir, labels
