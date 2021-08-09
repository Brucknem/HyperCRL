import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms


def create_encoder(device: torch.device) -> nn.Module:
    model_conv = torchvision.models.resnet18(pretrained=True)
    # TODO adjust ResNet18 last layers
    # model_conv.avgpool = nn.Sequential(
    #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
    #     nn.BatchNorm2d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
    #     nn.BatchNorm2d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
    #     nn.BatchNorm2d(512),
    # )
    # model_conv.fc = nn.Identity()
    print(model_conv)
    for param in model_conv.parameters():
        param.requires_grad = False
    model_conv.to(device)
    return model_conv


class SRL(nn.Module):

    def __init__(self, out_features):
        super(SRL, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.feature_extractor = create_encoder(self.device)

        self.representation_layers = nn.Sequential(
            # TODO adjust FC layers
            nn.Linear(in_features=self.feature_extractor.fc.out_features,
                      out_features=self.feature_extractor.fc.in_features),
            nn.Linear(in_features=self.feature_extractor.fc.in_features, out_features=out_features),
        )
        self.representation_layers.to(self.device)

    def forward(self, x):
        tensor = x
        if len(tensor.shape) == 3:
            tensor = self.normalize(x)
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        features = self.feature_extractor(tensor)
        # print(features)

        result = self.representation_layers(features)
        # print(result)

        result = result.cpu().detach().numpy()

        return result
