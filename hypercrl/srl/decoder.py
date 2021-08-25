import numpy as np
import torch.nn
import torchvision
from torch import nn


class SpatialBroadcastDecoder(torch.nn.Module):
    def __init__(self, in_features: int, w: int, h: int):
        super().__init__()
        self.in_features = in_features
        self.decoder = nn.Sequential(
            *([torch.nn.Conv2d(self.in_features + 2, self.in_features + 2, (1, 1)),
               torch.nn.LeakyReLU(0.1),
               torch.nn.BatchNorm2d(self.in_features + 2)] * 5),
            torch.nn.Conv2d(self.in_features + 2, 3, (1, 1)),
        )
        self.w = w
        self.h = h
        self.y = np.linspace(-1, 1, self.w)
        self.x = np.linspace(-1, 1, self.h)
        self.mesh = torch.tensor(np.meshgrid(self.x, self.y)).unsqueeze(1)

    def tile(self, z: torch.Tensor):
        batch_size = z.shape[0]
        z_b = [torch.tile(z[i], (self.w, self.h, 1)).permute(2, 0, 1) for i in range(batch_size)]
        mesh = self.mesh.to(z.device)
        z_sb = [torch.cat((i, mesh[0], mesh[1]), dim=0) for i in z_b]
        del mesh
        torch.cuda.empty_cache()
        z_sb = torch.stack(z_sb)
        return z_sb

    def forward(self, z: torch.Tensor):
        z_sb = self.tile(z)
        x = self.decoder(z_sb.float())
        return x
