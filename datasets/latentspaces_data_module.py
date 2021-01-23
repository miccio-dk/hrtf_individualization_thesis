import os
import random
import pytorch_lightning as pl
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from .latentspaces_dataset import LatentEarsHrtfDataset

class LatentSpacesDataModule(pl.LightningDataModule):
    def __init__(self, num_workers=4, batch_size=16, split=0.2, test_subjects=None, **kwargs):
        super().__init__()
        # store params
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.test_subjects = test_subjects
        self.args_ears, self.args_hrtf = kwargs['ears'], kwargs['hrtf']
        # setup dataset paths
        load_dotenv()
        path_basedir = os.getenv("HRTFI_DATA_BASEPATH")
        self.args_ears['z_path'] = os.path.join(path_basedir, self.args_ears['z_path'])
        self.args_ears['l_path'] = os.path.join(path_basedir, self.args_ears['l_path'])
        self.args_hrtf['z_path'] = os.path.join(path_basedir, self.args_hrtf['z_path'])
        self.args_hrtf['l_path'] = os.path.join(path_basedir, self.args_hrtf['l_path'])
        self.dataset_name = '{}{}'.format(
            'pca' if self.args_ears['n_pca'] else 'z',
            'pca' if self.args_hrtf['n_pca'] else 'z'
        )

    def setup(self, stage=None):
        # assign train/val split(s)
        if stage == 'fit' or stage is None:
            args_ears = dict(self.args_ears, skip_subjects=self.test_subjects)
            args_hrtf = dict(self.args_hrtf, skip_subjects=self.test_subjects)
            self.dataset = LatentEarsHrtfDataset(args_ears, args_hrtf)
            train_idx, val_idx = self._calc_splits(self.dataset, self.split)
            self.data_train = Subset(self.dataset, train_idx)
            self.data_val = Subset(self.dataset, val_idx)
            self.dims = self.data_train[0][0].shape
        # assign test split(s)
        if stage == 'test' or stage is None:
            args_ears = dict(self.args_ears, keep_subjects=self.test_subjects)
            args_hrtf = dict(self.args_hrtf, keep_subjects=self.test_subjects)
            self.data_test = LatentEarsHrtfDataset(args_ears, args_hrtf)
            if len(self.data_test) > 0:
                self.dims = getattr(self, 'dims', self.data_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _calc_splits(self, dataset, split):
        all_subjs = self.dataset.ds_hrtf.labels['subj'].unique()
        all_subjs = random.shuffle(all_subjs)
        idx = int(split * len(all_subjs))
        train_subjs = all_subjs[:idx]
        val_subjs = all_subjs[idx:]
        train_idx = self.dataset.ds_hrtf.labels['subj'].isin(train_subjs)
        val_idx = self.dataset.ds_hrtf.labels['subj'].isin(val_subjs)
        return train_idx, val_idx
