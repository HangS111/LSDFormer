import torch
import torch.nn as nn

from einops import rearrange, reduce
from ..modules import *


__all__ = ['CAA_DGCBS']

class CAA_DGCBS(nn.Module):
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )

        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))


    def forward(self, x):
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)

        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x