import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SRLDataLoader(pl.LightningDataModule):

    def __init__(self, srl_dataset):
        super().__init__()
        self.srl_dataset = srl_dataset

    def train_dataloader(self):
        return [DataLoader(), ]
