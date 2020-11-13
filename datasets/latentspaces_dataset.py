import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# generic latent space dataset
class LatentDataset(Dataset):
    def __init__(self, z_path, l_path, keep_subjects=None, skip_subjects=None, az_range=None, el_range=None, use_pca=False):
        self.z_path = z_path
        self.l_path = l_path
        self.keep_subjects = keep_subjects
        self.skip_subjects = skip_subjects
        self.az_range = az_range
        self.el_range = el_range
        self.use_pca = use_pca
        self.load_data()
        # TODO implement PCA

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            z = self.z[idx]
            labels = self.labels.iloc[idx]
        if isinstance(idx, tuple):
            subj, ear = idx
            idx = (self.labels['subj'] == subj) & (self.labels['ear'] == ear)
            z = self.z[idx]
            labels = self.labels[idx]
        sample = (z, labels)
        return sample

    def lbl_in_range(self, lbl, range_):
        if range_ is None:
            return
        if len(range_) == 2:
            keep_idx = (self.labels[lbl] >= range_[0]) & (self.labels[lbl] >= range_[1])
        else:
            keep_idx = self.labels[lbl].isin(range_)
        self.labels = self.labels[keep_idx]
        self.z = self.z[keep_idx.to_numpy()]

    def load_data(self):
        self.z = np.load(self.z_path)
        self.labels = pd.read_pickle(self.l_path)
        # filter by subj
        if self.keep_subjects:
            keep_idx = self.labels['subj'].isin(self.keep_subjects)
            self.labels = self.labels[keep_idx]
            self.z = self.z[keep_idx]
        if self.skip_subjects:
            skip_idx = ~self.labels['subj'].isin(self.skip_subjects)
            self.labels = self.labels[skip_idx]
            self.z = self.z[skip_idx]
        # filter by coord
        self.lbl_in_range('az', self.az_range)
        self.lbl_in_range('el', self.el_range)


# ears + hrtf latent spaces dataset
class LatentEarsHrtfDataset(Dataset):
    def __init__(self, data_cfg_ears, data_cfg_hrtf):
        self.ds_ears = LatentDataset(**data_cfg_ears)
        # only keep hrtf subjects with correspondance in ears dataset
        ears_subjs = set(self.ds_ears.labels['subj'].unique())
        if 'keep_subjects' in data_cfg_hrtf:
            hrtf_keep_subjects = ears_subjs.intersection(set(data_cfg_hrtf['keep_subjects']))
        if 'skip_subjects' in data_cfg_hrtf:
            hrtf_keep_subjects = ears_subjs.difference(set(data_cfg_hrtf['skip_subjects']))
        data_cfg_hrtf['keep_subjects'] = list(hrtf_keep_subjects)
        self.ds_hrtf = LatentDataset(**data_cfg_hrtf)

    def __len__(self):
        return len(self.ds_hrtf)

    def __getitem__(self, idx):
        # retrieve z_hrtf
        z_hrtf, l_hrtf = self.ds_hrtf[idx]
        # retrieve corresponding z_ears (same subj and ear)
        idx_ears = l_hrtf['subj'], l_hrtf['ear']
        z_ears, l_ears = self.ds_ears[idx_ears]
        # if multiple z_ears match, concatenate them
        if z_ears.ndim > 1:
            z_ears = np.concatenate(z_ears)
        assert isinstance(l_hrtf, pd.Series)
        sample = z_ears, z_hrtf, l_hrtf.to_dict()
        return sample
