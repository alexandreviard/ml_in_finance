from src.squelette_models import Head_Attention, MultiHead_Attention, SelfAttentionBlock
from src.utils import get_hierarchical_mask
import torch
import torch.nn as nn
    
class HMG_Intraday(nn.Module):
    def __init__(self, N=130, F=16, n_heads=4):
        super().__init__()
        mask_day  = get_hierarchical_mask(N, 26)
        mask_week = get_hierarchical_mask(N, 130)
        mask_none = None

        self.block1 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
            extra_mask=mask_day
        )
        self.block2 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
            extra_mask=mask_week
        )
        self.block3 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
            extra_mask=mask_none
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
    

class MG_Daily(nn.Module):
    def __init__(self, N=130, F=16, n_heads=4):
        super().__init__()

        self.block1 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
        )
        self.block2 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
        )
        self.block3 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            F=F,
            sigma_list=[5,10, 20, 40],
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x