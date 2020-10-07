import os
import requests
import pytorch_lightning as pl
from glob import glob
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
from .sofa_dataset import SofaDataset
from .data_transforms import ToHrtf, SpecEnv, ToTensor

class HrtfDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, nfft, feature=None, num_workers=4, batch_size=32, split=0.2, test_subjects=None, **kwargs):
        super().__init__()
        # store params
        self.nfft = nfft
        self.feature = feature
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.test_subjects = test_subjects
        self.ds_args = kwargs
        # select dataset to load
        self.dataset = None
        self.dataset_type = dataset_type
        self.dataset_path = {
            'viking': '/Users/miccio/OneDrive - Aalborg Universitet/viking_measurements/viking2/4_sofa',
            'hutubs': '/Users/miccio/work/aau/hutubs',
            'cipic': '/Users/miccio/work/aau/cipic'
        }.get(dataset_type)
        # setup transforms
        self.transforms = [ToHrtf(self.nfft), ToTensor()]
        if self.feature is not None:
            self.transforms.insert(1, SpecEnv(self.nfft, self.feature, cutoff=0.8))

    def prepare_data(self):
        # download cipic?
        if self.dataset_type == 'cipic':
            file_list = glob(os.path.join(self.dataset_path, '*.sofa'))
            if len(file_list) == 45:
                return
            ds_url = 'http://sofacoustics.org/data/database/cipic'
            os.makedirs(self.dataset_path, exist_ok=True)
            file_count = 0
            for i in range(166):
                file_url = f'{ds_url}/subject_{i:03}.sofa'
                file_path = f'{self.dataset_path}/subj_{i:03}.sofa'
                # check if already exists
                if os.path.exists(file_path):
                    pass
                # download file
                r = requests.get(file_url)
                # if download is successful, store
                if r.status_code == 200:
                    with open(file_path, 'wb') as fp:
                        fp.write(r.content)
                    file_count += 1

    def setup(self, stage=None):
        # assign train/val split(s)
        if stage == 'fit' or stage is None:
            self.dataset = SofaDataset(self.dataset_path, transform=Compose(self.transforms),
                                       skip_subjects=self.test_subjects, **self.ds_args)
            lengths = self._calc_splits(self.dataset, self.split)
            self.data_train, self.data_val = random_split(self.dataset, lengths)
            self.dims = self.data_train[0][0].shape
        # assign test split(s)
        if stage == 'test' or stage is None:
            self.data_test = SofaDataset(self.dataset_path, transform=Compose(self.transforms),
                                         keep_subjects=self.test_subjects, **self.ds_args)
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
