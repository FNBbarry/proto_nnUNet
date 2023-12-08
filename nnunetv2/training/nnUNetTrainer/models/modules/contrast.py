import torch
import torch.nn as nn
import torch.nn.functional as F


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))


class ProjectionHead_3D(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead_3D, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv3d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))

class ProjectionHead_3D_upscale(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead_3D_upscale, self).__init__()

        self.proj = nn.Sequential(
            nn.ConvTranspose3d(dim_in, dim_in // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(dim_in // 2, dim_in // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(dim_in // 4 , dim_in // 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(dim_in // 8 , proj_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return l2_normalize(self.proj(x))