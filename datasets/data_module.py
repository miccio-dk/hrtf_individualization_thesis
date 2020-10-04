import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
from .sofa_dataset import SofaDataset
from .data_transforms import ToHrtf, SpecEnv, ToTensor

class HrtfDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, nfft, feature=None, num_workers=4, batch_size=32, split=0.2):
        super().__init__()
        self.dataset_type = dataset_type
        self.nfft = nfft
        self.feature = feature
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.dataset = None

    def prepare_data(self):
        # select dataset to load
        dataset_path = {
            'viking': '/Users/miccio/OneDrive - Aalborg Universitet/viking_measurements/viking2/4_sofa',
            'hutubs': '',
            'cipic': ''
        }.get(self.dataset_type)
        # setup transforms
        transforms = []
        transforms.append(ToHrtf(self.nfft))
        if self.feature is not None:
            transforms.append(SpecEnv(self.nfft, self.feature, cutoff=0.8))
        transforms.append(ToTensor())
        # create dataset
        self.dataset = SofaDataset(dataset_path, transform=Compose(transforms))

    def setup(self, stage=None):
        # assign train/val split(s)
        if stage == 'fit' or stage is None:
            val_length = int(len(self.dataset) * self.split)
            lengths = [len(self.dataset) - val_length, val_length]
            self.data_train, self.data_val = random_split(self.dataset, lengths)
            self.dims = self.data_train[0][0].shape
        # assign test split(s)
        if stage == 'test' or stage is None:
            # TODO figure out a test dataset strategy
            self.data_test = None
            pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
