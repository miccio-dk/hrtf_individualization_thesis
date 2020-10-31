import os.path as osp
import numpy as np
import torch
import sofa
from glob import glob
from torch.utils.data import Dataset


# generic sofa dataset
class SofaDataset(Dataset):
    def __init__(self, data_path, transforms=None,
                 keep_subjects=None, skip_subjects=None,
                 az_range=None, el_range=None, ears=['L', 'R']):
        self.data_path = data_path
        self.transforms = transforms
        self.keep_subjects = keep_subjects
        self.skip_subjects = skip_subjects
        self.az_range = az_range
        self.el_range = el_range
        self.ears = ears
        self.hrirs = []
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
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    # works for filepath in the form '/path/to/file/subj_XXX.sofa'
    @staticmethod
    def path_to_subj(path):
        return osp.splitext(osp.basename(path))[0][5:]

    @staticmethod
    def get_label(subj, orient):
        return {
            'subj': subj,
            'ear': ['L', 'R'][int(orient[2])],
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
            system='spherical',
            angle_unit='degree')
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
        return hrirs, labels

    def lbl_in_range(self, val, range_):
        if range_ is None:
            return True
        if len(range_) == 2:
            return val >= range_[0] and val <= range_[1]
        else:
            return val in range_
        #raise Exception(f'range {range_} is not a valid range')

    def filter_data(self, hrirs, labels):
        new_hrirs = []
        new_labels = []
        for hrir, lbl in zip(hrirs, labels):
            if not self.lbl_in_range(lbl['az'], self.az_range):
                continue
            if not self.lbl_in_range(lbl['el'], self.el_range):
                continue
            if lbl['ear'] not in self.ears:
                continue
            new_hrirs.append(hrir)
            new_labels.append(lbl)
        new_hrirs = np.stack(new_hrirs)
        return new_hrirs, new_labels

    def load_data(self):
        subjects_paths = glob(osp.join(self.data_path, '*.sofa'))
        for i, path in enumerate(subjects_paths):
            # filter subject
            subj = SofaDataset.path_to_subj(path)
            if (self.keep_subjects is not None) and (subj not in self.keep_subjects):
                continue
            if (self.skip_subjects is not None) and (subj in self.skip_subjects):
                continue
            # load subject
            curr_hrirs, curr_labels = SofaDataset.load_sofa_subj(path)
            # filter data
            curr_hrirs, curr_labels = self.filter_data(curr_hrirs, curr_labels)
            # append data
            self.hrirs.append(curr_hrirs)
            self.labels.extend(curr_labels)
        # turn hrirs into matrix
        if self.hrirs:
            self.hrirs = np.concatenate(self.hrirs)
        else:
            self.hrirs = np.zeros((0, 0))
        #print(len(self.labels), self.hrirs.shape)
