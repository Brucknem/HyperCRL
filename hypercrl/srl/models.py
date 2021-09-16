import time

import numpy as np
import torch.nn
import torchvision
from torch import nn
from torchvision import transforms, models


class ResNet18Encoder:
    def __init__(self, mnet, hparams):
        self.feature_extractor = models.resnet18(pretrained=True)
        # self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()
        # self.feature_extractor.classifier[6] = torch.nn.Identity()
        for params in self.feature_extractor.parameters():
            params.requires_grad = False  #
        self.feature_extractor.eval()

        self.mnet = mnet
        self.image_shape = (-1, 3, hparams.vision_params.camera_widths, hparams.vision_params.camera_heights)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.param_shapes = self.mnet.param_shapes
        self.out_layer = torch.sigmoid
        self.out_var = hparams.vision_params.out_var

    def to(self, gpuid):
        self.feature_extractor.to(gpuid)
        self.mnet.to(gpuid)

    def train(self, train=True):
        self.feature_extractor.eval()
        self.mnet.train(train)

    def eval(self):
        self.train(False)

    def transform(self, x):
        x = x / 255.
        return self.transforms(x)

    def forward(self, x, weights):
        ts = time.time()
        if len(x.shape) == 1:
            x = x.reshape((-1, *x.shape))
        x = x.reshape(self.image_shape)
        x = self.transform(x)
        x = self.feature_extractor(x)
        x = self.mnet.forward(x, weights)
        # x = self.out_layer(x)
        # print(f'Encoding Time: {ts - time.time()} s')
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(MLP, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_dims, hidden_dims[0]),
            nn.LeakyReLU()
        )

        self.layers = []
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*self.layers)

        self.out_layer = nn.Linear(hidden_dims[-1], out_dims)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        x = self.out_layer(x)
        return x
