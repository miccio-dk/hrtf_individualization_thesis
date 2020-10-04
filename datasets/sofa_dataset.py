import os.path as osp
import numpy as np
import torch
import sofa
from glob import glob
from torch.utils.data import Dataset


# generic sofa dataset
class SofaDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.hrirs = None
        self.labels = []
        self.load_data()

    def __len__(self):
        return self.hrirs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        hrir = self.hrirs[idx]
        labels = self.labels[idx]
        sample = (hrir, labels)
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def path_to_subj(path):
        return osp.splitext(osp.basename(path))[0][5:]

    @staticmethod
    def get_label(subj, orient):
        return {
            'subj': subj,
            'ear': int(orient[2]),
            'az': orient[0],
            'el': orient[1],
        }

    @staticmethod
    def load_sofa_subj(path):
        subj = SofaDataset.path_to_subj(path)
        #print('loading subj', subj, '...')
        sofa_file = sofa.Database.open(path)
        # extract orientations and hrirs
        orients = sofa_file.Source.Position.get_values(
            system="spherical",
            angle_unit="degree")
        hrirs = sofa_file.Data.IR.get_values()
        n_orients = orients.shape[0]
        # "explode" along n_receiver
        hrirs = np.vstack([hrirs[:, 0], hrirs[:, 1]])
        orients = np.tile(orients, (2, 1))
        # use last column for ear channel
        orients[:n_orients, 2] = 0
        orients[n_orients:, 2] = 1
        # generate labels
        labels = [SofaDataset.get_label(subj, orient) for orient in orients]
        sofa_file.close()
        return labels, hrirs

    def load_data(self):
        subjects_paths = glob(osp.join(self.data_path, '*.sofa'))
        for i, path in enumerate(subjects_paths):
            curr_labels, curr_hrirs = SofaDataset.load_sofa_subj(path)
            # if first iteration, "discover" number of hrirs and create data structures
            if i == 0:
                self.hrirs = np.zeros((curr_hrirs.shape[0] * len(subjects_paths), curr_hrirs.shape[1]))
            # store data
            sl = slice(i * curr_hrirs.shape[0], (i + 1) * curr_hrirs.shape[0])
            self.hrirs[sl] = curr_hrirs
            self.labels.extend(curr_labels)
        #print(len(self.labels), self.hrirs.shape)
