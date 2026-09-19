"""Biased multi-kernel RBF MMD, valid for unequal nonempty batch sizes."""
import torch
from torch import nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul, self.kernel_num = kernel_mul, kernel_num
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2., kernel_num=5, fix_sigma=None):
        total = torch.cat((source, target)).float()
        distance = (total[:, None] - total[None, :]).square().sum(-1)
        n = len(total)
        bandwidth = (distance.detach().sum() / max(n * (n - 1), 1)
                     if fix_sigma is None else distance.new_tensor(fix_sigma))
        bandwidth = bandwidth.clamp_min(1e-8) / kernel_mul ** (kernel_num // 2)
        return sum(torch.exp(-distance / (bandwidth * kernel_mul ** i))
                   for i in range(kernel_num))

    def forward(self, source, target):
        if source.ndim != 2 or target.ndim != 2 or not len(source) or not len(target):
            raise ValueError('MMD requires nonempty [batch, features] tensors')
        k = self.guassian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
        n = len(source)
        return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()
