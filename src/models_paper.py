from src.squelette_models import Head_Attention, MultiHead_Attention, SelfAttentionBlock, LearnablePositionalEmbedding, TemporalAttention
from src.utils import get_hierarchical_mask
import torch
import torch.nn as nn
from typing import List, Optional
    
class HMG_Intraday(nn.Module):
    def __init__(self, N=130, F=16, n_heads=4, n_actifs:int=1, d_model:int=16, n_classes=2, sigma_list:List=[5,10, 20, 40]):
        super().__init__()
        mask_day  = get_hierarchical_mask(N, 26)
        mask_week = get_hierarchical_mask(N, 130)
        mask_none = None
    
        self.d_model = d_model
        self.positional_embedding = LearnablePositionalEmbedding(N, F)
        self.linear_input = nn.Linear(F, d_model)
        
        
        self.block1 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
            hierarchical_mask=mask_day
        )
        self.block2 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
            hierarchical_mask=mask_week
        )
        self.block3 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
            hierarchical_mask=mask_none
        )
        
        self.n_actifs = n_actifs
        self.linear_logits = nn.Linear(d_model, n_classes)

        

    def forward(self, x):
        x = torch.tanh((self.linear_input(self.positional_embedding(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.softmax(self.linear_logits(x))
        
        return x
    

class MG_Daily(nn.Module):
    def __init__(self, N=130, F=16, n_heads=4, n_actifs:int=1, d_model:int=16, n_classes=2, sigma_list:List=[5,10, 20, 40]):
        super().__init__()
        
        self.d_model = d_model
        self.positional_embedding = LearnablePositionalEmbedding(N, F)
        self.linear_input = nn.Linear(F, d_model)
        
        self.block1 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
        )
        self.block2 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
        )
        self.block3 = SelfAttentionBlock(
            n_heads=n_heads,
            N=N,
            d_model=d_model,
            sigma_list=sigma_list,
        )
        
        self.temporal_att = TemporalAttention(d_model=d_model, d_att=100)
        self.n_actifs = n_actifs
        self.linear_logits = nn.Linear(d_model, 1)
        
    def forward(self, x):

        x = torch.relu((self.linear_input(self.positional_embedding(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.temporal_att(x)
        
        return x
    
