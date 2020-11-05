import torch
import numpy as np
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn


def _conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    block = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
        ('bn2d', nn.BatchNorm2d(out_channels)),
        ('act', nn.LeakyReLU())
    ])
    return nn.Sequential(block)

def _conv_block_transp(in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_last=False):
    block = [
        ('upsamp2d', nn.UpsamplingNearest2d(scale_factor=2)),
        ('convtr2d', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
    ]
    if is_last:
        block.append(('act', nn.Sigmoid()))
        #block.append(('act', nn.LeakyReLU()))
    else:
        block.extend([
            ('bn2d', nn.BatchNorm2d(out_channels)),
            ('act', nn.LeakyReLU())
        ])
    return nn.Sequential(OrderedDict(block))

def _lin_block(in_size, out_size, dropout_rate=0.2):
    block = OrderedDict([
        ('lin', nn.Linear(in_size, out_size)),
        ('act', nn.LeakyReLU()),
        #('drop', nn.Dropout(dropout_rate))
    ])
    return nn.Sequential(block)

def _calc_output_shape(input_shape, model):
    in_tensor = torch.zeros(1, *input_shape)
    with torch.no_grad():
        out_tensot = model(in_tensor)
    return list(out_tensot.shape)[1:]


class ConvVAE(nn.Module):
    def __init__(self, input_shape, encoder_channels, latent_size, decoder_channels):
        super().__init__()
        assert len(input_shape) == 3
        assert type(encoder_channels) == list
        assert type(latent_size) == int
        assert type(decoder_channels) == list
        self.latent_size = latent_size
        self.enc = Encoder(input_shape, encoder_channels, latent_size)
        self.dec = Decoder(self.enc.conv_out_shape, decoder_channels, latent_size)

    def forward(self, x):
        means, log_var = self.enc(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.dec(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, input_shape, channels, latent_size):
        super().__init__()
        channels = [input_shape[0]] + channels
        self.conv_stack = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.conv_stack.add_module(name=f'block_{i}', module=_conv_block(in_channels, out_channels))
        self.conv_out_shape = _calc_output_shape(input_shape, self.conv_stack)
        linear_size = np.prod(self.conv_out_shape)
        self.linear_means = _lin_block(linear_size, latent_size)
        self.linear_log_var = _lin_block(linear_size, latent_size)

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, conv_out_shape, channels, latent_size):
        super().__init__()
        #channels = channels + channels[-1:]
        linear_size = np.prod(conv_out_shape)
        self.conv_out_shape = conv_out_shape
        self.linear_stack = _lin_block(latent_size, linear_size)
        self.conv_stack = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.conv_stack.add_module(name=f'block_{i}', module=_conv_block_transp(in_channels, out_channels))
        self.conv_stack.add_module(name='block_out', module=_conv_block_transp(channels[-1], 1, is_last=True))

    def forward(self, z):
        x = self.linear_stack(z)
        x = x.view(x.shape[0], *self.conv_out_shape)
        x = self.conv_stack(x)
        return x


# convolutional variational autoencoder
class VAECfg(pl.LightningModule):
    model_name = 'VAE_conv'

    def __init__(self, input_size, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.grad_freq = 50
        self.fig_freq = 10
        self.kl_coeff = cfg['kl_coeff']
        input_shape = [cfg['input_channels']] + input_size
        encoder_channels = cfg['encoder_channels']
        latent_size = cfg['latent_size']
        decoder_channels = cfg['decoder_channels']
        self.vae = ConvVAE(input_shape, encoder_channels, latent_size, decoder_channels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/vae_conv/small.json')
        return parser

    def loss_function(self, ear_true, ear_pred, means, log_var, z):
        # TODO replace binary cross entropy
        mse = torch.nn.functional.mse_loss(ear_pred, ear_true, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp()) * self.kl_coeff
        loss = (mse + kld) / ear_true.size(0)
        return mse, kld, loss

    def forward(self, x):
        return self.vae(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=50, cooldown=25),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        mse, kld, loss = self._shared_eval(batch, batch_idx)
        self.log('train_recon_loss', mse)
        self.log('train_kl', kld)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mse, kld, loss = self._shared_eval(batch, batch_idx)
        self.log('val_recon_loss', mse)
        self.log('val_kl', kld)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        mse, kld, loss = self._shared_eval(batch, batch_idx)
        self.log('test_recon_loss', mse)
        self.log('test_kl', kld)
        self.log('test_loss', loss)

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            img = self.get_pred_ear_figure(self.example_input_array, self.example_input_labels)
            self.logger.experiment.add_image('Valid/ears', img, self.current_epoch)

    def _shared_eval(self, batch, batch_idx):
        ear_true, labels = batch
        ear_pred, means, log_var, z = self.forward(ear_true)
        losses = self.loss_function(ear_true, ear_pred, means, log_var, z)
        return losses

    def get_pred_ear_figure(self, ear_true, labels, n_images=6):
        ear_true = ear_true.to(self.device)
        # run prediction
        self.eval()
        with torch.no_grad():
            ear_pred, means, log_var, z = self.forward(ear_true)
        self.train()
        img_true = torch.dstack(ear_true[:n_images].unbind())
        img_pred = torch.dstack(ear_pred[:n_images].unbind())
        img = torch.hstack((img_true, img_pred))
        return img
