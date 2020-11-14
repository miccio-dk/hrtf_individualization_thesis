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
        models[model_type] = ModelClass.load_from_checkpoint(model_ckpt_path)
        models[model_type].eval()
    print('### Models Loaded.')

    # load and process ear image
    print(f'### Loading and processing ear picture from {args.ear_path}...')
    img = Image.open(args.ear_path, 'r')
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
    ears = torch.stack([ear] * len(el_range))
    c = el_range.unsqueeze(-1)
    ears, c = ears.to(args.device), c.to(args.device)
    with torch.no_grad():
        # ear to z_ear
        _, z_ear, *_ = models['ears'](ears)
        # z_ear + c to z_hrtf
        x = torch.cat((z_ear, c), dim=-1)
        z_hrtf = models['latent'](x)
        # z_hrtf to hrtf
        hrtf = models['hrtf'].cvae.dec(z_hrtf, c)
    hrtf = hrtf.cpu().numpy()
    print(f'### Done predicting. Data shape: {hrtf.shape}')

    # generate figure
    if args.view:
        print('### Generating figure...')
        fig_output_path = os.path.splitext(args.output_path)[0] + '.png'
        n_cols = 6
        n_rows = math.ceil(len(el_range) / n_cols)
        ax_size = (4, 2.5)
        nfft = 256
        d = 1. / 48000
        f = rfftfreq(nfft, d)
        figsize = n_cols * ax_size[0], n_rows * ax_size[1]
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            if i < len(el_range):
                ax.plot(f, hrtf[i])
                ax.set_title(f'{el_range[i]}')
            else:
                ax.axis('off')
        fig.suptitle('PRTF along median plane')
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(fig_output_path)
        print(f'### Figure stored in {fig_output_path}')

    # store
    print(f'### Storing data in {args.output_path}...')
    sio.savemat(args.output_path, {'synthesized_hrtf': hrtf})
    print('### Done!')


if __name__ == '__main__':
    main()
