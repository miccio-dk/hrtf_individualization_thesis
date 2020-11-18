import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.fft import rfftfreq
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from lightning.models.cvae_dense_cfg import CVAECfg
from lightning.datasets.anthro_sofa_dataset import AnthroSofaDataset
from lightning.datasets.data_transforms import ToHrtf, ToDB, ToTensor


def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_ckpt_path', type=str)
    parser.add_argument('data_cfg_path', type=str)
    parser.add_argument('--nfft', default=256, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--sd_range', default=(500, 16000), type=int, nargs=2)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    # parse
    args = parser.parse_args()

    # load model
    print(f'### Loading model {CVAECfg.model_name} from {args.model_ckpt_path}...')
    model = CVAECfg.load_from_checkpoint(args.model_ckpt_path)
    model.to(args.device)
    model.eval()
    print('### Model Loaded.')

    # load data
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, data_cfg['dataset'])
    transforms = [ToHrtf(args.nfft), ToDB(), ToTensor()]

    print(f'### Loading data from {dataset_path}...')
    ds = AnthroSofaDataset(
        data_path=dataset_path,
        dataset_type=data_cfg['dataset'],
        keep_subjects=data_cfg['test_subjects'],
        transforms=Compose(transforms),
        az_range=data_cfg['az_range'],
        el_range=data_cfg['el_range'])
    dl = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    print(f'### Loaded {len(ds)} data points.')

    z = []
    lbl = []
    true = []
    pred = []
    print('### Predicting data...')
    for batch in tqdm(dl):
        resp_true, labels = batch
        c = torch.stack([labels[lbl] for lbl in model.c_labels], dim=-1).float()
        resp_true, c = resp_true.to(args.device), c.to(args.device)
        # run prediction
        with torch.no_grad():
            resp_pred, means, *_ = model(resp_true, c)
        means = means.to(args.device)
        labels = pd.DataFrame(labels)
        z.append(means)
        lbl.append(labels)
        true.append(resp_true)
        pred.append(resp_pred)

    # combine
    lbl = pd.concat(lbl)
    z = torch.cat(z).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    pred = torch.cat(pred).cpu().numpy()
    print(f'### Done converting. Data shapes: {z.shape} {lbl.shape} {pred.shape} {true.shape}')

    # calculate SD
    f = rfftfreq(args.nfft, d=1. / args.sr)
    idx = (f > args.sd_range[0]) & (f < args.sd_range[1])

    sd = np.sqrt(np.mean((true[:, idx] - pred[:, idx]) ** 2))
    print(f'### Spectral distortion (dB) = {sd}')


if __name__ == '__main__':
    main()
