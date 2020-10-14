import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
from .sofa_dataset import SofaDataset
from .data_transforms import ToHrtf, SpecEnv, ToTensor
from .utils import download_sofa

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
            'cipic': '/Users/miccio/work/aau/cipic',
            'ari_inear': '/Users/miccio/work/aau/ari_inear'
        }.get(dataset_type)
        # setup transforms
        self.transforms = [ToHrtf(self.nfft), ToTensor()]
        if self.feature is not None:
            self.transforms.insert(1, SpecEnv(self.nfft, self.feature, cutoff=0.8))

    def prepare_data(self):
        # download cipic
        if self.dataset_type == 'cipic':
            ds_url_file = 'cipic/subject_{:03}.sofa'
            expected_count = 45
            file_count = download_sofa(self.dataset_path, ds_url_file, expected_count)
            assert file_count == expected_count
        elif self.dataset_type == 'ari_inear':
            ds_url_file = 'ari/hrtf_nh{}.sofa'
            expected_count = 97
            file_count = download_sofa(self.dataset_path, ds_url_file, expected_count)
            assert file_count == expected_count
        elif self.dataset_type == 'hutubs':
            ds_url_file = 'hutubs/pp{}_HRIRs_measured.sofa'
            expected_count = 96
            file_count = download_sofa(self.dataset_path, ds_url_file, expected_count)
            assert file_count == expected_count

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
