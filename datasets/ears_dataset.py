import os
import random
import json
import os.path as osp
from glob import glob
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, RandomHorizontalFlip
from torch.utils.data import Dataset, ConcatDataset
from .data_transforms import AddRandomNoise


# generic ear dataset
class EarsDataset(Dataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96), augmentations=None, mode='rgb', no_labels=False):
        self.data_path = data_path
        self.no_labels = no_labels
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
        # flip images in AMI
        if 'feature' in labels and labels['feature'] != 'back':
            img = RandomHorizontalFlip(1.)(img)
        # apply augmentations
        if self.augmentations:
            augmentation = random.choice(self.augmentations)
            if augmentation:
                img = augmentation(img)
        if self.no_labels:
            sample = (img, labels['subj'])
        else:
            sample = (img, labels)
        return sample


# AMI ear pictures dataset
class AmiDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', no_labels=False, features=None):
        super(AmiDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode, no_labels=no_labels)
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


# HUTUBS ear renderings dataset
class HutubsEarsDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', no_labels=False, az_range=None, el_range=None):
        super(HutubsEarsDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode, no_labels=no_labels)
        self.az_range = az_range
        self.el_range = el_range
        self.load_data()

    # works for filepath in the form '/basepath/ear/ele_azi_0_0/ppX_3DheadMesh.png'
    @staticmethod
    def parse_filename(path):
        subj = int(osp.basename(path).split('_')[0][2:])
        el, az = osp.basename(osp.dirname(path)).split('_')[:2]
        ear = osp.basename(osp.dirname(osp.dirname(path)))[0].upper()
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


# AWE ears dataset
class AweDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', no_labels=False):
        super(AweDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode, no_labels=no_labels)
        self.load_data()

    # works for filepath in the form '/basepath/XXX/YYY.png'
    @staticmethod
    def generate_labels(path, item):
        subj = osp.basename(osp.dirname(path))
        ear = item['d'].upper()
        return {
            'subj': subj,
            'ear': ear,
        }

    def load_data(self):
        dir_paths = glob(osp.join(self.data_path, '*/'))
        for dir_path in dir_paths:
            # load annotations
            with open(osp.join(dir_path, 'annotations.json'), 'r') as fp:
                ann = json.load(fp)
            # load individual images
            for k, item in ann['data'].items():
                # extract infos
                path = osp.join(dir_path, item['file'])
                label = AweDataset.generate_labels(path, item)
                # filter by subject
                if not self.is_subject_included(label['subj']):
                    continue
                # load image
                img = Image.open(path, 'r').convert('RGB')
                # mirror if right
                if label['ear'] == 'R':
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # append data
                self.ears.append(img)
                self.labels.append(label)


# IITD ears dataset
class IitdDataset(EarsDataset):
    def __init__(self, data_path, keep_subjects=None, skip_subjects=None, img_size=(96, 96),
                 augmentations=None, mode='rgb', no_labels=False):
        super(IitdDataset, self).__init__(
            data_path, keep_subjects=keep_subjects, skip_subjects=skip_subjects,
            img_size=img_size, augmentations=augmentations, mode=mode, no_labels=no_labels)
        self.load_data()

    # works for filepath in the form '/basepath/raw//XXX_n.png'
    @staticmethod
    def parse_filename(path):
        subj = osp.splitext(osp.basename(path))[0].split('_')[0]
        return {
            'subj': subj,
        }

    def load_data(self):
        file_paths = glob(osp.join(self.data_path, 'raw', '*.bmp'))
        for path in file_paths:
            # extract infos
            label = IitdDataset.parse_filename(path)
            # filter by subject
            if not self.is_subject_included(label['subj']):
                continue
            # load image and preprocess
            img = Image.open(path, 'r').convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)
            # append data
            self.ears.append(img)
            self.labels.append(label)


# combination of the other datasets
class CombinedEarsDataset(Dataset):
    def __init__(self, data_path=None, keep_subjects={}, skip_subjects={}, img_size=(96, 96), augmentations=None, mode='rgb', datasets_configs={}):
        load_dotenv()
        path_basedir = os.getenv("HRTFI_DATA_BASEPATH")
        DS = {
            'ami_ears': AmiDataset,
            'hutubs_ears': HutubsEarsDataset,
            'awe_ears': AweDataset,
            'iitd_ears': IitdDataset
        }
        # init and merge datasets
        datasets = []
        for dataset_type, kwargs in datasets_configs.items():
            dataset_path = os.path.join(path_basedir, dataset_type)
            _skip_subjects = skip_subjects.get(dataset_type)
            _keep_subjects = keep_subjects.get(dataset_type)
            ds = DS[dataset_type](
                data_path=dataset_path,
                keep_subjects=_keep_subjects,
                skip_subjects=_skip_subjects,
                img_size=img_size,
                augmentations=augmentations,
                mode=mode,
                no_labels=True,
                **datasets_configs[dataset_type])
            datasets.append(ds)
        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
