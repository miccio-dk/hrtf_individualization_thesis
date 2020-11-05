import json
import torch
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from models.endtoend_cfg import EndToEndCfg
from models.cvae_dense_cfg import CVAECfg
from datasets.hrtf_data_module import HrtfDataModule

def cli_main():
    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('model_type', type=str)
    parser.add_argument('data_cfg_path', type=str)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--test', action='store_false', dest='train')
    parser.add_argument('--gpus', default=None)
    # model args
    # TODO use correct class or make 'add_model_specific_args' a method of model parents
    parser = EndToEndCfg.add_model_specific_args(parser)
    # parse
    args = parser.parse_args()

    # ensure reproducibility
    if args.dev:
        pl.seed_everything(1337)

    # pick model
    ModelClass = {
        'ete': EndToEndCfg,
        'cvae_dense': CVAECfg,
    }.get(args.model_type)

    # load data configs
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)
    # init data loader
    dm = HrtfDataModule(
        dataset_type=data_cfg['dataset'],
        nfft=args.nfft,
        feature=data_cfg['feature'],
        use_db=data_cfg['use_db'],
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_subjects=data_cfg['test_subjects'],
        az_range=data_cfg['az_range'],
        el_range=data_cfg['el_range'])

    if args.train:
        # load model configs
        with open(args.model_cfg_path, 'r') as fp:
            model_cfg = json.load(fp)
        # init model
        model = ModelClass(nfft=args.nfft, cfg=model_cfg)

        # pass first batch of validation data for plotting
        # TODO this part changes depending on the model
        if args.model_type == 'cvae_dense':
            dm.prepare_data()
            dm.setup(stage=None)
            val_resp, val_labels = next(iter(dm.val_dataloader()))
            val_c = torch.stack([val_labels[lbl] for lbl in model_cfg['labels']], dim=-1).float()
            model.example_input_array = (val_resp, val_c)
            model.example_input_labels = pd.DataFrame(val_labels)

    # logger
    log_name = '{}_{}_{}'.format(model.model_name, data_cfg['dataset'], data_cfg['feature'])
    logger = TensorBoardLogger('logs', name=log_name)

    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint = ModelCheckpoint(monitor='val_loss')

    # training
    trainer = pl.Trainer(
        weights_summary='full',
        max_epochs=args.max_epochs,
        limit_train_batches=0.1 if args.dev else 1.0,
        limit_val_batches=0.3 if args.dev else 1.0,
        profiler=args.dev,
        callbacks=[early_stop, lr_monitor, checkpoint],
        resume_from_checkpoint=args.resume_path,
        terminate_on_nan=False,
        gradient_clip_val=0.5,
        logger=logger,
        deterministic=args.dev,
        gpus=args.gpus)

    if args.train:
        trainer.fit(model, dm)
    else:
        # TODO implement
        trainer.test()


if __name__ == '__main__':
    cli_main()
