import torch
import torch.nn as nn


def weights_init(m):
    if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d}:
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.0)


pass