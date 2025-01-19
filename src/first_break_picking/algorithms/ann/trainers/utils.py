import os
import torch


def detach(x):
    return x.cpu().detach().numpy()


def clip_grad(m):
    [p.grad.data.clamp_(-1, 1) for p in m.parameters() if p.grad is not None]


def l1_reg(w):
    return torch.mean(torch.abs(w))


def clear():
    _ = os.system("clear" if os.name == "posix" else "cls")


pass