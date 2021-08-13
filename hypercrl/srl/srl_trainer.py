import os

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import multiprocessing

from hypercrl.srl import SRL


class SRLTrainer:
    def __init__(self, srl: SRL, save_dir="srl_logs", name="srl", batch_size=128):
        self.srl = srl
        self.name = name
        self.root_dir = os.path.join("../../../", save_dir)
        self.model_checkpoint_name = "latest.ckpt"
        self.cpu_cores = multiprocessing.cpu_count()
        self.callbacks = [
            EarlyStopping('losses/total_loss'),
        ]
        self.logger = TensorBoardLogger(self.root_dir, name=self.name, default_hp_metric=False)
        self.batch_size = batch_size

    def train(self, srl_dataset):
        dataloader = DataLoader(srl_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_cores)

        checkpoint_path = os.path.join(self.root_dir, self.name, self.model_checkpoint_name)
        log_every_n_steps = int(self.batch_size / 10)

        trainer = pl.Trainer(gpus=1, logger=self.logger, log_every_n_steps=log_every_n_steps, callbacks=self.callbacks)
        trainer.fit(self.srl, dataloader)
        trainer.save_checkpoint(checkpoint_path)
