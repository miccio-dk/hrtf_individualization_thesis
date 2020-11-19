import os
import torch
from argparse import ArgumentParser
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, ToPILImage
from ..models.vae_conv_cfg import VAECfg
from ..models.vae_resnet_cfg import ResNetVAECfg
from ..models.vae_incept_cfg import InceptionVAECfg


def main():
    # load env
    load_dotenv()
    path_basedir = os.getenv("HRTFI_DATA_BASEPATH")
    default_output_path = os.path.join(path_basedir, 'output.png')

    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_type', type=str)
    parser.add_argument('model_ckpt_path', type=str)
    parser.add_argument('ear_path', type=str)
    parser.add_argument('--output_path', default=default_output_path, type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument('--img_channels', default=1, type=int)
    parser.add_argument('--view', action='store_true')
    # parse
    args = parser.parse_args()

    # pick models
    ModelClass = {
        'vae_conv': VAECfg,
        'vae_resnet': ResNetVAECfg,
        'vae_incept': InceptionVAECfg
    }.get(args.model_type)
    # load model
    print(f'### Loading model {ModelClass.model_name} from {args.model_ckpt_path}...')
    model = ModelClass.load_from_checkpoint(args.model_ckpt_path)
    model.to(args.device)
    model.eval()
    print('### Model Loaded.')

    # load and process ear image
    print(f'### Loading and processing ear picture from {args.ear_path}...')
    img = Image.open(args.ear_path, 'r').convert('RGB')
    transforms = Compose([
        Resize(args.img_size),
        ToTensor(),
        Grayscale(args.img_channels)
    ])
    ear = transforms(img)
    print('### Done loading and processing.')

    # predict datapoints
    print('### Predicting data...')
    ear_true = ear.unsqueeze(0).to(args.device)
    with torch.no_grad():
        ear_pred, z_ear, *_ = model(ear_true)
    ear_pred = ear_pred.cpu()[0]
    z = z_ear.cpu()[0].numpy()
    print(f'### Done predicting. Latent space: \n{z}')

    # store
    print(f'### Storing reconstructed ear in {args.output_path}...')
    img = ToPILImage()(ear_pred)
    img.save(args.output_path)
    print('### Done!')


if __name__ == '__main__':
    main()
