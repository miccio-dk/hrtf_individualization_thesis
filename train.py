import json
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from models.endtoend_cfg import EndToEndCfg
from models.cvae_dense_cfg import CVAECfg
from datasets.data_module import HrtfDataModule

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
    # model args
    parser = EndToEndCfg.add_model_specific_args(parser)
    # parse
    args = parser.parse_args()

    # ensure reproducibility
    if args.dev:
        pl.seed_everything(1337)

    # load configs
    with open(args.model_cfg_path, 'r') as fp:
        model_cfg = json.load(fp)
    with open(args.data_cfg_path, 'r') as fp:
        data_cfg = json.load(fp)

    # data loaders
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
    dm.prepare_data()
    dm.setup(stage=None)

    # model
    model_class = {
        'ete': EndToEndCfg,
        'cvae_dense': CVAECfg
    }.get(args.model_type)
    model = model_class(nfft=args.nfft, cfg=model_cfg)

    # pass first batch of validation data for plotting
    val_resp, val_labels = next(iter(dm.val_dataloader()))
    val_c = val_labels['el'].unsqueeze(-1).float()
    model.example_input_array = (val_resp, val_c)
    model.example_input_labels = pd.DataFrame(val_labels)

    # logger
    log_name = '{}_{}_{}'.format(model.model_name, data_cfg['dataset'], data_cfg['feature'])
    logger = TensorBoardLogger('logs', name=log_name)

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
