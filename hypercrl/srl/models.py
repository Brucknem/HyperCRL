import time

import numpy as np
import torch.nn
import torchvision
from torch import nn
from torchvision import transforms, models

from hypercrl.srl.utils import stack_sin_cos


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def layer_to_name(layer):
    if isinstance(layer, nn.ReLU):
        return ["relu"]
    if isinstance(layer, nn.Linear):
        return ["linear"]
    if isinstance(layer, nn.LeakyReLU):
        return ["leaky_relu"]
    if isinstance(layer, nn.Sequential):
        return [layer_to_name(inner) for inner in layer]
    return ["unknown"]


def layer_names(net):
    names = []
    for c in net.children():
        names += layer_to_name(c)
    names = flatten(names)
    names = [f'{i}_{name}' for i, name in enumerate(names)]
    return names


class MLP(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(MLP, self).__init__()
        act_fn = nn.ReLU
        self.in_layer = nn.Sequential(
            nn.Linear(in_dims, hidden_dims[0]),
            act_fn()
        )

        self.layers = []
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                act_fn()
            ))
        self.layers = nn.Sequential(*self.layers)
        self.out_layer = nn.Sequential(nn.Linear(hidden_dims[-1], out_dims))

        self.apply(init_weights)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        x = self.out_layer(x)
        return x


class ResNet18Encoder(torch.nn.Module):
    def __init__(self, mnet, vparams):
        super().__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        # self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()
        # self.feature_extractor.classifier[6] = torch.nn.Identity()
        for params in self.feature_extractor.parameters():
            params.requires_grad = False  #
        self.feature_extractor.eval()

        self.mnet = mnet
        self.image_shape = (-1, 3, vparams.camera_widths, vparams.camera_heights)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.out_layer = torch.sigmoid
        self.out_var = vparams.encoder_model.out_var
        self.weight_names = mnet.weight_names
        self.add_sin_cos_to_state = vparams.add_sin_cos_to_state

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

    def parameters(self, recurse=True):
        return self.mnet.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.mnet.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self, x, weights):
        ts = time.time()
        if len(x.shape) == 1:
            x = x.reshape((-1, *x.shape))
        x = x.reshape(self.image_shape)
        x = self.transform(x)
        x = self.feature_extractor(x)
        x = self.mnet.forward(x)
        if self.add_sin_cos_to_state:
            x = stack_sin_cos(x)
        # x = self.out_layer(x)
        # print(f'Encoding Time: {ts - time.time()} s')
        return x


class ResNet18EncoderHnet(ResNet18Encoder):
    def __init__(self, mnet, vparams):
        super(ResNet18EncoderHnet, self).__init__(mnet, vparams)
        self.param_shapes = self.mnet.param_shapes

    def forward(self, x, weights):
        ts = time.time()
        if len(x.shape) == 1:
            x = x.reshape((-1, *x.shape))
        x = x.reshape(self.image_shape)
        x = self.transform(x)
        x = self.feature_extractor(x)
        x = self.mnet.forward(x, weights)
        if self.add_sin_cos_to_state:
            x = stack_sin_cos(x)
            # x = self.out_layer(x)
        # print(f'Encoding Time: {ts - time.time()} s')
        return x


class StaticFeatureExtractor(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()
        self.device = 'cpu'

    def forward(self, x):
        if x is None:
            return x

        with torch.no_grad():
            x = self.transform(x)
            if x.device != self.device:
                x = x.to(self.device)

            if len(x.shape) == 3:
                x = x.unsqueeze(dim=0)
            x = self.feature_extractor(x)
            return x

    def to(self, device):
        self.feature_extractor.to(device)
        self.device = device
        return self


class MyMLP(nn.Module):
    def __init__(self, in_dims, out_dims, h_dims, hidden_layers=1, use_batch_norm=True, dropout=-1.,
                 act_fn=nn.LeakyReLU()):
        super().__init__()

        if isinstance(h_dims, int):
            h_dims = [h_dims] * hidden_layers

        dims = [in_dims, *h_dims, out_dims]

        self.bn = nn.ModuleList([nn.BatchNorm1d(dims[i]) for i in range(1, len(dims) - 1)]) if use_batch_norm else None
        self.linear = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = nn.Dropout(dropout) if 0 <= dropout < 1 else None
        self.act_fn = act_fn

        self.weight_names = list(dict(self.named_parameters()).keys())

    def forward(self, x, weights=None):
        for i, layer in enumerate(self.linear):
            x = layer(x)
            x = self.act_fn(x)

            if i != len(self.linear) - 1:
                if self.bn is not None:
                    x = self.bn[i](x)

                if self.dropout is not None:
                    x = self.dropout(x)

        return x


if __name__ == "__main__":
    x = torch.ones((10, 7))
    print(x)

    mlp = MyMLP(7, 2, 100, 2, use_batch_norm=True, dropout=0.5)
    print(mlp)
    mlp.eval()
    result = mlp(x)

    print(result)

    params = list(mlp.parameters())
    named_params = dict(mlp.named_parameters())
    # print(params)
    for name, param in mlp.named_parameters():
        if param.requires_grad:
            print(name)

    print("Yeet")
