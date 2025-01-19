import torch
import torch.nn as nn
from torchsummary import summary
from .device import device


def nonlinearity():
    return nn.ELU(inplace=True)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,  32,  (32, 8), stride=(4, 2)),
            nonlinearity(),
            nn.Conv2d(32, 64,  (16, 4), stride=(4, 2)),
            nonlinearity(),
            nn.Conv2d(64, 64,  (2, 2),  stride=(2, 2)),
            nonlinearity(),
            nn.Conv2d(64, 128, (8, 2),  stride=(2, 2)),
            nonlinearity(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (128, 7, 2)),
            nn.ConvTranspose2d(128, 64, (9, 3), stride=(2, 2)),
            nonlinearity(),
            nn.ConvTranspose2d(64, 64, (3, 2), stride=(2, 2)),
            nonlinearity(),
            nn.ConvTranspose2d(64, 32, (16, 4), stride=(4, 2)),
            nonlinearity(),
            nn.ConvTranspose2d(32, 1, (32, 8), stride=(4, 2)),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return self.model(x)[:, :, :751, :48]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        encoded = self.enc(x)
        return self.dec(encoded)


model = Model().to(device)
loss = torch.nn.MSELoss()


summary(model, input_size=(1, 751, 48)),


pass