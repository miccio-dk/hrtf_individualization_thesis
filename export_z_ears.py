import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.vae_conv_cfg import VAECfg
from models.vae_resnet_cfg import ResNetVAECfg
from models.vae_incept_cfg import InceptionVAECfg
from datasets.ears_dataset import AmiDataset, HutubsEarsDataset


def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_type', type=str)
    parser.add_argument('model_ckpt_path', type=str)
    parser.add_argument('--data_cfg_path', default='./configs/data/ears/hutubs.json', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--output_path', default=path_basedir, type=str)
    # parse
    args = parser.parse_args()

    # pick model
    ModelClass = {
        'vae_conv': VAECfg,
        'vae_resnet': ResNetVAECfg,
        'vae_incept': InceptionVAECfg
    }.get(args.model_type)
    # load model
    print(f'### Loading model {ModelClass.model_name} from {args.model_ckpt_path}...')
    model = ModelClass.load_from_checkpoint(args.model_ckpt_path)
    model.eval()
    print('### Model Loaded.')

    # load data config
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, data_cfg['dataset_type'])

    # pick dataset
    DatasetClass = {
        'ami_ears': AmiDataset,
        'hutubs_ears': HutubsEarsDataset
    }.get(data_cfg['dataset_type'])
    # load data
    print(f'### Loading data from {dataset_path}...')
    data_cfg.pop('dataset_type', None)
    data_cfg.pop('test_subjects', None)
    data_cfg.pop('augmentations', None)
    ds = DatasetClass(
        data_path=dataset_path,
        **data_cfg)
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
        ear_true, labels = batch
        ear_true = ear_true.to(args.device)
        # run prediction
        with torch.no_grad():
            ear_pred, means, *_ = model(ear_true)
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
    np.save(os.path.join(args.output_path, 'z_ears.npy'), z)
    lbl.to_pickle(os.path.join(args.output_path, 'l_ears.pkl'))
    print('### Done!')


if __name__ == '__main__':
    main()
