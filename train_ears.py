import json
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from models.vae_conv_cfg import VAECfg
from models.vae_resnet_cfg import ResNetVAECfg
from datasets.ears_data_module import EarsDataModule

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
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--test', action='store_false', dest='train')
    parser.add_argument('--gpus', default=None)
    # model args
    parser = VAECfg.add_model_specific_args(parser)
    # parse
    args = parser.parse_args()

    # ensure reproducibility
    if args.dev:
        pl.seed_everything(1337)

    # pick model
    ModelClass = {
        'vae_conv': VAECfg,
        'vae_resnet': ResNetVAECfg
    }.get(args.model_type)

    # load data configs
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)
    # init data loader
    dm = EarsDataModule(
        dataset_type=data_cfg['dataset'],
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_subjects=data_cfg['test_subjects'],
        img_size=data_cfg['img_size'],
        features=data_cfg['features'],
        augmentations=data_cfg['augmentations'],
        mode=data_cfg['mode'])

    if args.train:
        # load model configs
        with open(args.model_cfg_path, 'r') as fp:
            model_cfg = json.load(fp)
        # init model
        model = ModelClass(input_size=data_cfg['img_size'], cfg=model_cfg)

        # pass first batch of validation data for plotting
        dm.prepare_data()
        dm.setup(stage=None)
        val_img, val_labels = next(iter(dm.val_dataloader()))
        model.example_input_array = val_img
        model.example_input_labels = pd.DataFrame(val_labels)

    # logger
    log_name = '{}_{}'.format(model.model_name, data_cfg['dataset'])
    logger = TensorBoardLogger('logs', name=log_name, log_graph=False)

    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=100)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint = ModelCheckpoint(monitor='val_loss')

    # training
    trainer = pl.Trainer(
        weights_summary='full',
        max_epochs=args.max_epochs,
        limit_train_batches=0.1 if args.dev else 1.0,
        limit_val_batches=0.3 if args.dev else 1.0,
        profiler=args.dev,
        callbacks=[lr_monitor, checkpoint],
        resume_from_checkpoint=args.resume_path,
        terminate_on_nan=False,
        gradient_clip_val=0.5,
        logger=logger,
        deterministic=args.dev,
        gpus=args.gpus)

    if args.train:
        trainer.fit(model, dm)
    else:
        trainer.test()


if __name__ == '__main__':
    cli_main()
