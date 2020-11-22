import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.fft import rfftfreq
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from models.vae_conv_cfg import VAECfg
from models.vae_resnet_cfg import ResNetVAECfg
from models.vae_incept_cfg import InceptionVAECfg
from models.dnn_cfg import DNNCfg
from models.cvae_dense_cfg import CVAECfg
from datasets.ears_dataset import HutubsEarsDataset
from datasets.sofa_dataset import SofaDataset
from datasets.data_transforms import ToHrtf, ToDB
from datasets.data_transforms import ToTensor as ToHrtfTensor


def create_range(_range):
    _range[1] += 1
    if len(_range) == 3:
        _range = torch.arange(*_range)
    elif len(_range) == 2:
        _range = torch.arange(*_range, 10)
    return _range

def get_matching_resps_true(dataset_path, subj, transforms, az_range, el_range, ear):
    hrtf_ds = SofaDataset(
        data_path=dataset_path,
        keep_subjects=[subj],
        transforms=Compose(transforms),
        az_range=az_range,
        el_range=el_range,
        ears=[ear])
    labels = pd.DataFrame([l for h, l in hrtf_ds])
    idxs = labels.sort_values('el').index
    hrtf = torch.stack([hrtf_ds[i][0] for i in idxs])
    return hrtf

def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('cfg_path', type=str)
    parser.add_argument('ear_data_cfg_path', type=str)
    parser.add_argument('hrtf_data_cfg_path', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--nfft', default=256, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--sd_range', default=(500, 16000), type=int, nargs=2)
    parser.add_argument('--batch_size', default=32, type=int)
    # parse
    args = parser.parse_args()

    # load configs
    with open(args.cfg_path, 'r') as fp:
        cfg = json.load(fp)
    img_size = cfg['ears']['img_size']
    img_channels = cfg['ears']['img_channels']

    # pick models
    EarsModelClass = {
        'vae_conv': VAECfg,
        'vae_resnet': ResNetVAECfg,
        'vae_incept': InceptionVAECfg
    }.get(cfg['ears']['model_type'])
    LatentModelClass = {
        'dnn': DNNCfg
    }.get(cfg['latent']['model_type'])
    HrtfModelClass = {
        'cvae_dense': CVAECfg,
    }.get(cfg['hrtf']['model_type'])

    # load models
    models = {}
    for ModelClass, model_type in zip([EarsModelClass, LatentModelClass, HrtfModelClass], ['ears', 'latent', 'hrtf']):
        model_ckpt_path = cfg[model_type]['model_ckpt_path']
        print(f'### Loading model {ModelClass.model_name} from {model_ckpt_path}...')
        model = ModelClass.load_from_checkpoint(model_ckpt_path)
        model.to(args.device)
        model.eval()
        models[model_type] = model
    print('### Models Loaded.')

    # load ears data
    with open(args.ear_data_cfg_path, 'r') as fp:
        ear_data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, ear_data_cfg['dataset_type'])
    transforms = Compose([
        Resize(img_size),
        ToTensor(),
        Grayscale(img_channels)
    ])
    print(f'### Loading ears data from {dataset_path}...')
    ears_ds = HutubsEarsDataset(
        data_path=dataset_path,
        keep_subjects=ear_data_cfg['test_subjects'],
        mode=ear_data_cfg['mode'],
        az_range=[0],
        el_range=[0])
    print(f'### Loaded {len(ears_ds)} data points.')

    # calculate elevation range
    el_range = cfg['el_range']
    if el_range:
        el_range = create_range(el_range)
    # calculate azimuth range
    az_range = cfg['az_range']
    if az_range:
        az_range = create_range(az_range)
    # create c tensor
    if el_range is not None and az_range is not None:
        c = torch.cartesian_prod(el_range, az_range)
    elif el_range is not None:
        c = el_range.unsqueeze(-1)
        az_range = [0]
    elif az_range is not None:
        c = az_range.unsqueeze(-1)
        el_range = [0]
    c = c.to(args.device)

    # hrtf data configs
    print(f'### Loading hrtf data from {dataset_path}...')
    with open(args.hrtf_data_cfg_path, 'r') as fp:
        hrtf_data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, hrtf_data_cfg['dataset'])
    transforms = [ToHrtf(args.nfft), ToDB(), ToHrtfTensor()]

    # run prediction
    resps_true = []
    resps_pred = []
    print('### Predicting data...')
    for batch in tqdm(ears_ds):
        ear, labels = batch
        ear = ear.unsqueeze(0).to(args.device)
        with torch.no_grad():
            # ear to z_ear
            _, z_ear, *_ = models['ears'](ear)
            z_ears = z_ear.repeat(c.shape[0], 1)
            # z_ear + c to z_hrtf
            x = torch.cat((z_ears, c), dim=-1)
            z_hrtf = models['latent'](x)
            # z_hrtf to hrtf
            resp_pred = models['hrtf'].cvae.dec(z_hrtf, c)
        resp_true = get_matching_resps_true(dataset_path, labels['subj'], transforms, az_range, el_range, labels['ear'])
        # store results
        resps_true.append(resp_true)
        resps_pred.append(resp_pred)

    # combine
    resps_true = torch.cat(resps_true).cpu().numpy()
    resps_pred = torch.cat(resps_pred).cpu().numpy()
    print(f'### Done converting. Data shapes: {resps_pred.shape} {resps_true.shape}')

    # calculate SD
    f = rfftfreq(args.nfft, d=1. / args.sr)
    idx = (f > args.sd_range[0]) & (f < args.sd_range[1])

    sd = np.sqrt(np.mean((resps_true[:, idx] - resps_pred[:, idx]) ** 2))
    print(f'### Spectral distortion (dB) = {sd}')


if __name__ == '__main__':
    main()
