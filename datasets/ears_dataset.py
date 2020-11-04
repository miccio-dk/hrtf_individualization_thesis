import random
import os.path as osp
from glob import glob
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, RandomHorizontalFlip
from torch.utils.data import Dataset
from .data_transforms import AddRandomNoise


# generic ear dataset
class EarsDataset(Dataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96), augmentations=None, mode='rgb'):
        self.data_path = data_path
        # setup necessary transforms
        self.transforms = [
            Resize(img_size),
            ToTensor(),
        ]
        if mode == 'bw':
            self.transforms.append(Grayscale())
        elif mode == 'bw3':
            self.transforms.append(Grayscale(num_output_channels=3))
        self.transforms = Compose(self.transforms)
        # setup optional augmentations
        self.augmentations = []
        if augmentations:
            self.augmentations = [AddRandomNoise(mode=mode) for mode in augmentations]
            # chance for no augmentation
            self.augmentations.append(None)
        # other params
        self.keep_subjects = keep_subjects
        self.skip_subjects = skip_subjects
        self.ears = []
        self.labels = []

    def is_subject_included(self, subj):
        if (self.keep_subjects is not None) and (subj not in self.keep_subjects):
            return False
        if (self.skip_subjects is not None) and (subj in self.skip_subjects):
            return False
        return True

    def __len__(self):
        return len(self.ears)

    def __getitem__(self, idx):
        assert type(idx) == int
        ear = self.ears[idx]
        labels = self.labels[idx]
        # apply transforms
        if self.transforms:
            img = self.transforms(ear)
        if 'feature' in labels and labels['feature'] != 'back':
            img = RandomHorizontalFlip(1.)(img)
        # apply augmentations
        if self.augmentations:
            augmentation = random.choice(self.augmentations)
            if augmentation:
                img = augmentation(img)
        sample = (img, labels)
        return sample


# ami ear pictures dataset
class AmiDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', features=None):
        super(AmiDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode)
        self.features = features
        self.load_data()

    # works for filepath in the form '/path/to/file/XXX_feature_ear.jpg'
    @staticmethod
    def parse_filename(path):
        subj, feature, _ = osp.splitext(osp.basename(path))[0].split('_')
        return {
            'subj': subj,
            'feature': feature
        }

    def load_data(self):
        file_paths = glob(osp.join(self.data_path, '*.jpg'))
        for i, path in enumerate(file_paths):
            # extract infos
            label = AmiDataset.parse_filename(path)
            # filter by subject
            if not self.is_subject_included(label['subj']):
                continue
            # filter by features
            if self.features and label['feature'] not in self.features:
                continue
            # load image
            img = Image.open(path, 'r')
            # append data
            self.ears.append(img)
            self.labels.append(label)


# hutubs ear renderings dataset
class HutubsEarsDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', az_range=None, el_range=None):
        super(HutubsEarsDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode)
        self.az_range = az_range
        self.el_range = el_range
        self.load_data()

    # works for filepath in the form '/basepath/ear/ele_azi_0_0/ppX_3DheadMesh.png'
    @staticmethod
    def parse_filename(path):
        subj = int(osp.basename(path).split('_')[0][2:])
        el, az = osp.basename(osp.dirname(path)).split('_')[:2]
        ear = osp.basename(osp.dirname(osp.dirname(path)))
        return {
            'subj': f'{subj:03}',
            'ear': ear,
            'el': float(el),
            'az': float(az),
        }

    @staticmethod
    def lbl_in_range(val, range_):
        if range_ is None:
            return True
        if len(range_) == 2:
            return val >= range_[0] and val <= range_[1]
        else:
            return val in range_

    def load_data(self):
        file_paths = glob(osp.join(self.data_path, '**/*.png'), recursive=True)
        for i, path in enumerate(file_paths):
            # extract infos
            label = HutubsEarsDataset.parse_filename(path)
            # filter by subject
            if not self.is_subject_included(label['subj']):
                continue
            # filter by features
            if not self.lbl_in_range(label['az'], self.az_range):
                continue
            if not self.lbl_in_range(label['el'], self.el_range):
                continue
            # load image
            img = Image.open(path, 'r').convert('RGB')
            # append data
            self.ears.append(img)
            self.labels.append(label)
