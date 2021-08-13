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

from hypercrl.srl.robotic_priors import Slowness, Variability


class SRL(pl.LightningModule):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.representation_layers = nn.Sequential(
            # TODO adjust FC layers
            # nn.Linear(in_features=self.feature_extractor.fc.out_features,
            #           out_features=self.feature_extractor.fc.in_features),
            nn.Linear(in_features=self.feature_extractor.fc.out_features, out_features=out_features),
        )
        self.representation_layers.to(self.device)

        self.slowness_prior = Slowness()
        self.variability_prior = Variability()

    def forward(self, x):
        features = self.feature_extractor(x)
        result = self.representation_layers(features)
        # result = result.cpu().detach().numpy()
        return result

    def log_loss(self, name: str, value: any):
        self.log("losses/" + name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def project_forward(self, batch):
        observations = batch["similar_states"]["observations"] + batch["dissimilar_states"]["observations"]
        result = []
        for observation in observations:
            with torch.no_grad():
                features = self.feature_extractor(observation)
            representation = self.representation_layers(features)
            result.append(representation)
        return result

    def training_step(self, batch, batch_idx):
        observations = self.project_forward(batch)

        total_loss = 0
        slowness_loss = self.slowness_prior(observations[0], observations[1])
        total_loss += slowness_loss

        variability_loss = self.variability_prior(observations[0], observations[1]) + self.variability_prior(
            observations[2], observations[3])
        total_loss += variability_loss

        self.log_loss("total_loss", total_loss)
        self.log_loss("slowness_loss", slowness_loss)
        self.log_loss("variability_loss", variability_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
