import os
import json
import warnings
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.fft import rfftfreq
from torchvision.transforms import Compose
from models.vae_conv_cfg import VAECfg
from models.vae_resnet_cfg import ResNetVAECfg
from models.vae_incept_cfg import InceptionVAECfg
from models.dnn_cfg import DNNCfg
from models.cvae_dense_cfg import CVAECfg
from datasets.ears_dataset import HutubsEarsDataset
from datasets.sofa_dataset import SofaDataset
from datasets.latentspaces_dataset import LatentDataset
from datasets.latentspaces_data_module import LatentSpacesDataModule
from datasets.data_transforms import ToHrtf, ToDB
from datasets.data_transforms import ToTensor as ToHrtfTensor
warnings.simplefilter("ignore", UserWarning)


def sd(resps_true, resps_pred, idx):
    _sd = np.sqrt(np.mean((resps_true[idx] - resps_pred[idx]) ** 2))
    return _sd

def sd_minimum(resps_true, resps_pred, idx, offs_range=[-5, 5], step=0.1):
    offsets = np.arange(*offs_range, step)
    bs = resps_true.shape[0]
    _sd = [[sd(resps_true[i], resps_pred[i] + offs, idx) for offs in offsets] for i in range(bs)]
    _sd = np.array(_sd)
    _sd = np.amin(_sd, axis=1)
    return _sd.mean()

def matching_resp_true(ds, labels, subj, ear, c):
    subj_labels = labels[(labels['ear'] == ear) & (labels['subj'] == subj)]
    hrtf_list = []
    for el, az in c.cpu().numpy():
        if ear == 'R' and az == -180:
            az = 180
        idx = subj_labels.index[(subj_labels['el'] == el) & (subj_labels['az'] == az)][0]
        hrtf_list.append(ds[idx][0])
    hrtf = torch.stack(hrtf_list)
    return hrtf

def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('eval_cfg_path', type=str)
    parser.add_argument('ear_cfg_path', type=str)
    parser.add_argument('hrtf_cfg_path', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--nfft', default=256, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--sd_range', default=(500, 16000), type=int, nargs=2)
    parser.add_argument('--z_cfg_path', default=None, type=str)
    # parse
    args = parser.parse_args()

    # load configs
    with open(args.eval_cfg_path, 'r') as fp:
        cfg = json.load(fp)

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
    with open(args.ear_cfg_path, 'r') as fp:
        ear_data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, ear_data_cfg['dataset_type'])
    print(f'### Loading ears data from {dataset_path}...')
    ears_ds = HutubsEarsDataset(
        data_path=dataset_path,
        keep_subjects=ear_data_cfg['test_subjects'],
        mode=ear_data_cfg['mode'],
        az_range=[0],
        el_range=[0],
        img_size=ear_data_cfg['img_size'])
    print(f'### Loaded {len(ears_ds)} ears data points.')

    # load hrtf data
    print(f'### Loading hrtf data from {dataset_path}...')
    with open(args.hrtf_cfg_path, 'r') as fp:
        hrtf_data_cfg = json.load(fp)
    dataset_path = os.path.join(path_basedir, hrtf_data_cfg['dataset'])
    transforms = [ToHrtf(args.nfft), ToDB(), ToHrtfTensor()]
    hrtf_ds = SofaDataset(
        data_path=dataset_path,
        keep_subjects=hrtf_data_cfg['test_subjects'],
        transforms=Compose(transforms),
        az_range=hrtf_data_cfg['az_range'],
        el_range=hrtf_data_cfg['el_range'])
    hrtf_ds_labels = pd.DataFrame([l for h, l in hrtf_ds])
    hrtf_labels = hrtf_ds_labels[hrtf_ds_labels['subj'] == hrtf_ds_labels['subj'][0]]
    hrtf_labels = hrtf_labels[hrtf_labels['ear'] == 'L']
    print(f'### Loaded {len(hrtf_ds)} hrtf data points ({len(hrtf_labels) * 2} per subject).')

    # if needed, load pca
    n_pca_hrtf = cfg['n_pca']['hrtf']
    if n_pca_hrtf:
        try:
            # load z_hrtf cfg
            with open(args.z_cfg_path, 'r') as fp:
                z_cfg = json.load(fp)
            # init data loader
            dm = LatentSpacesDataModule(
                **z_cfg,
                num_workers=0,
                batch_size=32)
            dm.setup(stage=None)
            z_train = dm.dataset.ds_hrtf.z
        except Exception as e:
            print(f'Error loading z_hrtf config file {args.z_cfg_path}')
            raise e
        # calculate loadings and scores on train set
        z_pc_train, scaler, pca = LatentDataset.generate_pca(z_train, n_pca=n_pca_hrtf)

    # create c tensor
    c = [torch.tensor(hrtf_labels[lbl].values) for lbl in models['hrtf'].c_labels]
    c = torch.stack(c, dim=-1).float()
    c = c.to(args.device)

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
            # if pca (hrtf), z_pc_hrtf to z_hrtf
            if n_pca_hrtf:
                z_hrtf = z_hrtf.cpu()
                z_hrtf = LatentDataset.pca_inverse_transform(z_hrtf, scaler, pca)
                z_hrtf = z_hrtf.to(args.device)
            # z_hrtf to hrtf
            resp_pred = models['hrtf'].cvae.dec(z_hrtf, c)
        resp_true = matching_resp_true(hrtf_ds, hrtf_ds_labels, labels['subj'], labels['ear'], c)
        assert len(resp_true) == len(resp_pred)
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
    sd = sd_minimum(resps_true, resps_pred, idx)
    print(f'### Spectral distortion (dB) = {sd}')


if __name__ == '__main__':
    main()
