import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from hypercrl.srl.robotic_priors import SlownessPrior, VariabilityPrior, ProportionalityPrior, RepeatabilityPrior, \
    CausalityPrior, ReferencePointPrior


class SRL(pl.LightningModule):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.encoder = torchvision.models.resnet18(pretrained=True)

        for params in self.encoder.parameters():
            params.requires_grad = False

        self.encoder.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.Linear(in_features=512, out_features=out_features)
        )
        self.encoder.fc.requires_grad = True

        self.slowness_prior = SlownessPrior()
        self.variability_prior = VariabilityPrior()
        self.proportionality_prior = ProportionalityPrior()
        self.repeatability_prior = RepeatabilityPrior()
        self.causality_prior = CausalityPrior()
        self.reference_point_prior = ReferencePointPrior()

    def forward(self, x):
        result = self.encoder(x)
        return result

    def log_loss(self, prefix: str, name: str, value: any):
        title = name
        if prefix != '':
            title = prefix + "/" + name
        self.log(title, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        state = self.encoder(batch['observations'][0][0])
        next_state = self.encoder(batch['observations'][0][1])
        other_state = self.encoder(batch['observations'][1][0])
        other_next_state = self.encoder(batch['observations'][1][1])

        total_loss = 0
        slowness_loss = self.slowness_prior(state, next_state) + self.slowness_prior(other_state, other_next_state)
        total_loss += slowness_loss

        variability_loss = self.variability_prior(state, other_state) + \
                           self.variability_prior(state, other_next_state) + \
                           self.variability_prior(other_state, next_state) + \
                           self.variability_prior(other_next_state, next_state)
        total_loss += variability_loss

        scale = 10
        proportionality_loss = self.proportionality_prior(state, next_state, other_state, other_next_state)
        total_loss += scale * proportionality_loss

        repeatability_loss = self.repeatability_prior(state, next_state, other_state, other_next_state)
        total_loss += scale * repeatability_loss

        causality_loss = self.causality_prior(state, other_state, batch["rewards"][0][0], batch["rewards"][1][0])
        total_loss += scale * causality_loss

        self.log_loss("robotic_priors", "all", total_loss)
        self.log_loss("robotic_priors", "slowness_loss", slowness_loss)
        self.log_loss("robotic_priors", "variability_loss", variability_loss)
        self.log_loss("robotic_priors", "proportionality_loss", proportionality_loss)
        self.log_loss("robotic_priors", "repeatability_loss", repeatability_loss)
        self.log_loss("robotic_priors", "causality_loss", causality_loss)

        self.log_loss("", "_total_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
