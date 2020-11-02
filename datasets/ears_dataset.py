import os.path as osp
from glob import glob
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, RandomHorizontalFlip
from torch.utils.data import Dataset


# generic sofa dataset
class EarsDataset(Dataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96), features=None, mode='rgb'):
        self.data_path = data_path
        self.transforms = [
            Resize(img_size),
            ToTensor(),
        ]
        if mode == 'bw':
            self.transforms.append(Grayscale())
        elif mode == 'bw3':
            self.transforms.append(Grayscale(num_output_channels=3))
        self.transforms = Compose(self.transforms)

        self.keep_subjects = keep_subjects
        self.skip_subjects = skip_subjects
        self.features = features
        self.ears = []
        self.labels = []
        self.load_data()

    def __len__(self):
        return len(self.ears)

    def __getitem__(self, idx):
        assert type(idx) == int
        ear = self.ears[idx]
        labels = self.labels[idx]
        if self.transforms:
            img = self.transforms(ear)
        if labels['feature'] == 'back':
            img = RandomHorizontalFlip(1.)(img)
        sample = (img, labels)
        return sample

    # works for filepath in the form '/path/to/file/XXX_feature_ear.jpg'
    @staticmethod
    def parse_filename(path):
        subj, feature, _ = osp.splitext(osp.basename(path))[0].split('_')
        return subj, feature

    def load_data(self):
        file_paths = glob(osp.join(self.data_path, '*.jpg'))
        for i, path in enumerate(file_paths):
            # extract infos
            subj, feature = EarsDataset.parse_filename(path)
            # filter data
            if (self.keep_subjects is not None) and (subj not in self.keep_subjects):
                continue
            if (self.skip_subjects is not None) and (subj in self.skip_subjects):
                continue
            if self.features and feature not in self.features:
                continue
            # load image
            img = Image.open(path, 'r')
            # append data
            self.ears.append(img)
            self.labels.append({
                'subj': subj,
                'feature': feature
            })
