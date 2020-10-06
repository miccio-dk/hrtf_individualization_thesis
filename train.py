import json
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from models.model_cfg import AutoEncoderCfg
from datasets.data_module import HrtfDataModule

def cli_main():
    # args
    parser = ArgumentParser()
    # trainer args
    parser.add_argument('--dataset', default='viking', type=str)
    parser.add_argument('--feature', default=None, type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # model args
    parser = AutoEncoderCfg.add_model_specific_args(parser)
    # parse
    args = parser.parse_args()

    # ensure reproducibility
    if args.dev:
        pl.seed_everything(1337)

    # data loaders
    dm = HrtfDataModule(
        dataset_type=args.dataset,
        nfft=args.nfft,
        feature=args.feature,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_subjects=['kemarLast', 'A', 'B'],
        az_range=[0],
        el_range=(-45, 45))
    dm.prepare_data()
    dm.setup(stage=None)

    # load model configs
    with open(args.model_cfg_path, 'r') as fp:
        cfg = json.load(fp)

    # model
    model = AutoEncoderCfg(
        nfft=args.nfft,
        cfg=cfg,
        log_on_batch=args.log_on_batch)
    # pass first batch of validation data for plotting
    val_batch, val_labels = next(iter(dm.test_dataloader()))
    model.example_input_array = val_batch
    model.example_input_labels = pd.DataFrame(val_labels)

    # logger
    logger = TensorBoardLogger('logs', name=model.__class__.__name__)

    # training
    trainer = pl.Trainer(
        weights_summary='full',
        max_epochs=args.max_epochs,
        limit_train_batches=0.1 if args.dev else 1.0,
        limit_val_batches=0.1 if args.dev else 1.0,
        profiler=args.dev,
        callbacks=[LearningRateLogger(logging_interval='epoch')],
        #default_root_dir=args.root_path,
        resume_from_checkpoint=args.resume_path,
        early_stop_callback=False and True,
        terminate_on_nan=False,
        gradient_clip_val=0.5,
        logger=logger,
        deterministic=args.dev)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
