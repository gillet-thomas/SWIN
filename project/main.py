import time
import yaml
import warnings
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from module.utils.data_module import ADNISwiFTDataset
from module.pl_classifier import LitClassifier


def cli_main():
    config = yaml.safe_load(open("config.yaml"))
    Classifier = LitClassifier
    Dataset = ADNISwiFTDataset

    item = Dataset(config)[0]
    print(item)

    timepoint = time.strftime("%Y%m%d-%H%M%S")
    default_root_dir = f"output/{config['project_name']}/{timepoint}"
    data_module = Dataset(config)
    pl.seed_everything(config['seed'])

    # ------------ logger -------------
    wandb_logger = WandbLogger(
        project="fMRI2Vec",
        name=config['project_name'],
        config=config,
        mode="online"
    )
    
    # ------------ callbacks -------------
    # callback for classification task
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_root_dir,
        monitor="valid_acc",
        filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
        save_last=True,
        mode="max",
    )

    # ------------ trainer -------------
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = pl.Trainer(
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        max_epochs=config['max_epochs'],
        default_root_dir=default_root_dir,
        precision=config['precision'],
    )

    # ------------ model -------------
    model = Classifier(data_module = data_module, config=config) 

    # ------------ run -------------
    if config['test_only']:
        trainer.test(model, datamodule=data_module, ckpt_path=config['test_ckpt_path']) # dataloaders=data_module
    else:
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cli_main()
