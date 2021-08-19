import os

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import multiprocessing

from hypercrl.srl import SRL, SRLDataSet


class SRLTrainer:
    def __init__(self, srl: SRL, save_dir="srl_logs", name="srl"):
        self.srl = srl
        self.name = name
        self.root_dir = os.path.join("../../../", save_dir)
        self.model_checkpoint_name = ''
        self.last_epoch = 0

        self.cpu_cores = multiprocessing.cpu_count()
        self.callbacks = [
            # EarlyStopping('_total_loss', min_delta=0.0, patience=5),
        ]
        self.logger = TensorBoardLogger(self.root_dir, name=self.name, default_hp_metric=False)

    def train(self, srl_dataset: SRLDataSet, min_epochs: int = 15, max_epochs: int = 100, batch_size=128):
        dataloader = DataLoader(srl_dataset, batch_size=batch_size, shuffle=True, num_workers=self.cpu_cores)

        log_every_n_steps = 10

        params = {
            'gpus': 1,
            'logger': self.logger,
            'log_every_n_steps': log_every_n_steps,
            'callbacks': self.callbacks,
            'max_epochs': self.last_epoch + max_epochs,
            'min_epochs': self.last_epoch + min_epochs
        }

        if os.path.isfile(self.model_checkpoint_name):
            params['resume_from_checkpoint'] = self.model_checkpoint_name

        trainer = pl.Trainer(**params)
        trainer.fit(self.srl, dataloader)
        self.model_checkpoint_name = os.path.join(trainer.checkpoint_callback.dirpath, "latest.ckpt")
        trainer.save_checkpoint(self.model_checkpoint_name)
        self.last_epoch = trainer.current_epoch
