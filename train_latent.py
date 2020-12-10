import json
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from models.dnn_cfg import DNNCfg
from datasets.latentspaces_data_module import LatentSpacesDataModule

def cli_main():
    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_type', type=str)
    parser.add_argument('data_cfg_path', type=str)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--test_ckpt_path', default=None, type=str)
    parser.add_argument('--gpus', default=None)
    # model args
    parser = DNNCfg.add_model_specific_args(parser)
    # parse
    args = parser.parse_args()
    is_training = args.test_ckpt_path is None

    # ensure reproducibility
    if args.dev:
        pl.seed_everything(1337)

    # pick model
    ModelClass = {
        'dnn': DNNCfg
    }.get(args.model_type)

    # load data configs
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)
    # init data loader
    dm = LatentSpacesDataModule(
        **data_cfg,
        num_workers=args.num_workers,
        batch_size=args.batch_size)

    # logger
    log_name = '{}_{}'.format(ModelClass.model_name, dm.dataset_name)
    logger = TensorBoardLogger('logs', name=log_name, log_graph=False)

    if is_training:
        # get first batch of validation data
        dm.setup(stage=None)
        print(f'### Data len (train val test): {len(dm.data_train)} {len(dm.data_val)} {len(dm.data_test)}')
        z_ears, z_hrtf_true, labels = next(iter(dm.val_dataloader()))

        # load model configs
        with open(args.model_cfg_path, 'r') as fp:
            model_cfg = json.load(fp)
        model_cfg['z_ears_size'] = z_ears.shape[1]
        model_cfg['z_hrtf_size'] = z_hrtf_true.shape[1]
        # init model
        model = ModelClass(cfg=model_cfg)

        # pass data batch to model for plotting
        model.example_labels_array = labels
        c = torch.stack([labels[lbl] for lbl in model_cfg['c']], dim=-1).float()
        model.example_input_array = torch.cat((z_ears, c), dim=-1)
        model.example_output_array = z_hrtf_true

        # callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=args.patience)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # training
        trainer = pl.Trainer(
            weights_summary='full',
            max_epochs=args.max_epochs,
            limit_train_batches=0.1 if args.dev else 1.0,
            limit_val_batches=0.3 if args.dev else 1.0,
            profiler=args.dev,
            callbacks=[early_stop, lr_monitor],
            resume_from_checkpoint=args.resume_path,
            terminate_on_nan=False,
            gradient_clip_val=0.5,
            logger=logger,
            deterministic=args.dev,
            gpus=args.gpus)
        trainer.fit(model=model, datamodule=dm)
    else:
        trainer = pl.Trainer(
            logger=logger,
            weights_summary='full',
            profiler=args.dev,
            deterministic=args.dev,
            gpus=args.gpus)
        model = ModelClass.load_from_checkpoint(
            checkpoint_path=args.test_ckpt_path,
            map_location=None)
        trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
