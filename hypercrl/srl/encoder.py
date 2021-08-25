import numpy as np
import torch.nn
import torchvision
from torch import nn
from torchvision import transforms


class ResNet18Encoder(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.encoder = torchvision.models.resnet18(pretrained=True)

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224))
        ])

        for params in self.encoder.parameters():
            params.requires_grad = False

        self.encoder.fc = nn.Sequential(
            nn.Linear(in_features=self.encoder.fc.in_features, out_features=out_features)
        )
        self.encoder.fc.requires_grad = True

        self.hidden2mu = nn.Linear(out_features, out_features)
        self.hidden2log_var = nn.Linear(out_features, out_features)

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def forward(self, x):
        x = self.transform(x)
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        hidden = self.reparametrize(mu, log_var)
        return {"z": hidden, "mu": mu, "log_var": log_var}

    def transform(self, x: torch.Tensor):
        return self.transform(x)
