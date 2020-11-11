import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models.cvae_dense_cfg import CVAECfg
from datasets.anthro_sofa_dataset import AnthroSofaDataset
from datasets.data_transforms import ToHrtf, ToDB, ToTensor


def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_ckpt_path', type=str)
    parser.add_argument('--data_cfg_path', default='./configs/data/hrtf/hutubs_full.json', type=str)
    parser.add_argument('--nfft', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--output_path', default=path_basedir, type=str)
    # parse
    args = parser.parse_args()

    # load model
    print(f'### Loading model from {args.model_ckpt_path}...')
    model = CVAECfg.load_from_checkpoint(args.model_ckpt_path)
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

    # calculate each datapoint
    z = []
    lbl = []
    print('### Converting data...')
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

    # combine
    z = torch.cat(z).numpy()
    lbl = pd.concat(lbl)
    print(f'### Done converting. Data shapes: {z.shape} {lbl.shape}')

    # store
    print(f'### Storing data in {args.output_path}...')
    np.save(os.path.join(args.output_path, 'z_hrtf.npy'), z)
    lbl.to_pickle(os.path.join(args.output_path, 'l_hrtf.pkl'))
    print('### Done!')


if __name__ == '__main__':
    main()
