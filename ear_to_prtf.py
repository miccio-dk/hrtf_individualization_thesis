import os
import math
import json
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dotenv import load_dotenv
from PIL import Image
from scipy.fft import rfftfreq
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from models.vae_conv_cfg import VAECfg
from models.vae_resnet_cfg import ResNetVAECfg
from models.vae_incept_cfg import InceptionVAECfg
from models.dnn_cfg import DNNCfg
from models.cvae_dense_cfg import CVAECfg


def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")
    default_output_path = os.path.join(path_basedir, 'output.mat')

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('cfg_path', type=str)
    parser.add_argument('ear_path', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--nfft', default=256, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--output_path', default=default_output_path, type=str)
    parser.add_argument('--view', action='store_true')
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

    # load and process ear image
    print(f'### Loading and processing ear picture from {args.ear_path}...')
    img = Image.open(args.ear_path, 'r').convert('RGB')
    transforms = Compose([
        Resize(img_size),
        ToTensor(),
        Grayscale(img_channels)
    ])
    ear = transforms(img)
    print('### Done loading and processing.')

    # calculate range
    # TODO support azimuth too
    el_range = cfg['el_range']
    el_range[1] += 1
    if len(el_range) == 3:
        el_range = torch.arange(*el_range)
    elif len(el_range) == 2:
        el_range = torch.arange(*el_range, 10)

    # predict datapoints
    print('### Predicting data...')
    ear = ear.unsqueeze(-1)
    c = el_range.unsqueeze(-1)
    ear, c = ear.to(args.device), c.to(args.device)
    with torch.no_grad():
        # ear to z_ear
        _, z_ear, *_ = models['ears'](ear)
        z_ears = torch.stack([z_ear] * len(el_range))
        # z_ear + c to z_hrtf
        x = torch.cat((z_ears, c), dim=-1)
        z_hrtf = models['latent'](x)
        # z_hrtf to hrtf
        hrtf = models['hrtf'].cvae.dec(z_hrtf, c)
    hrtf = hrtf.cpu().numpy()
    print(f'### Done predicting. Data shape: {hrtf.shape}')

    # generate figure
    if args.view:
        print('### Generating figure...')
        output_path_resps = os.path.splitext(args.output_path)[0] + '_resps.png'
        output_path_surf = os.path.splitext(args.output_path)[0] + '_surf.png'
        f = rfftfreq(args.nfft, d=1. / args.sr)
        # make first figure (individual responses)
        n_cols = 6
        n_rows = math.ceil(len(el_range) / n_cols)
        figsize = n_cols * 4, n_rows * 2.4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            if i < len(el_range):
                ax.plot(f, hrtf[i])
                ax.set_title(f'{el_range[i]}')
            else:
                ax.axis('off')
        fig.suptitle('PRTF along median plane')
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(output_path_resps)
        # make second figure (surface plot)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        extent = [f[0], f[-1], el_range[0], el_range[-1]]
        im = ax.imshow(hrtf, extent=extent, aspect='auto', vmin=-80, vmax=20, cmap='viridis')
        ax.set_title('PRTF along median plane')
        ax.set_ylabel('Elevation')
        ax.set_ylabel('Frequency [kHz]')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path_surf)
        print(f'### Figure stored in {output_path_resps} and {output_path_surf}')

    # store
    print(f'### Storing data in {args.output_path}...')
    sio.savemat(args.output_path, {'synthesized_hrtf': hrtf})
    print('### Done!')


if __name__ == '__main__':
    main()
