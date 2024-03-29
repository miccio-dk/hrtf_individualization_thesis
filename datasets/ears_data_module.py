import os
import pytorch_lightning as pl
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from .ears_dataset import AmiDataset, HutubsEarsDataset, AweDataset, IitdDataset, CombinedEarsDataset

class EarsDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, num_workers=4, batch_size=16, split=0.1, test_subjects=None, **kwargs):
        super().__init__()
        load_dotenv()
        # store params
        self.dataset_type = dataset_type
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.test_subjects = test_subjects
        self.ds_args = kwargs
        self.path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    def setup(self, stage=None):
        # pick dataset
        DS = {
            'ami_ears': AmiDataset,
            'hutubs_ears': HutubsEarsDataset,
            'hutubs_edges': HutubsEarsDataset,
            'hutubs_depth': HutubsEarsDataset,
            'awe_ears': AweDataset,
            'iitd_ears': IitdDataset,
            'combined': CombinedEarsDataset
        }.get(self.dataset_type)
        dataset_path = os.path.join(self.path_basedir, self.dataset_type)
        # assign train/val split(s)
        if stage == 'fit' or stage is None:
            self.dataset = DS(data_path=dataset_path, skip_subjects=self.test_subjects, **self.ds_args)
            lengths = self._calc_splits(self.dataset, self.split)
            self.data_train, self.data_val = random_split(self.dataset, lengths)
            self.dims = self.data_train[0][0].shape
        # assign test split(s)
        if stage == 'test' or stage is None:
            self.data_test = DS(data_path=dataset_path, keep_subjects=self.test_subjects, **self.ds_args)
            if len(self.data_test) > 0:
                self.dims = getattr(self, 'dims', self.data_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _calc_splits(self, dataset, split):
        data_len = len(dataset)
        val_len = int(data_len * split)
        lengths = [data_len - val_len, val_len]
        return lengths
