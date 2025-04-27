import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np
import math

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, N: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, N, F) * 0.01)

    def forward(self, x):
        x = x + self.pos_embedding
        return x
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, N: int, d_model: int):
        super().__init__()
        self.d_model = d_model

        pos = torch.arange(N).unsqueeze(1)  # (N, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # (d_model//2,)

        pe = torch.zeros(1, N, d_model)  # (1, N, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[0, :, 1::2] = torch.cos(pos * div)
        else:
            pe[0, :, 1::2] = torch.cos(pos * div)[:, :-1]  # on ignore la dernière colonne de cos si impair

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Head_Attention(nn.Module):
    """
    Classe qui définit une tête d'attention (self-attention) avec un éventuel prior gaussien.
    N : la taille de la window considéré (en horizon temporel)
    d_model : la dimension du modèle après la projection de (NxF) vers (Nxd_model)
    """
    def __init__(self, d_model: int, head_size: int, sigma: Optional[float] = None, pdrop: float = 0.1):
        
        super().__init__()
        self.head_size = head_size
        self.key_linear = nn.Linear(d_model, head_size)
        self.query_linear = nn.Linear(d_model, head_size)
        self.value_linear = nn.Linear(d_model, head_size)
        self.register_buffer('scale', torch.tensor(1.0 / math.sqrt(head_size)))
        self.sigma = sigma
        
        self.attn_dropout = nn.Dropout(pdrop)
        self.out_proj_drop = nn.Dropout(pdrop)

    def forward(self, x, hierarchical_mask=None):
        B, N, d_model = x.shape
        key = self.key_linear(x) # (Nxd_model) -> (NxH)
        query = self.query_linear(x) # (Nxd_model) -> (NxH)
        value = self.value_linear(x) # (Nxd_model) -> (NxH)
        score = (query @ key.transpose(-2, -1)) * self.scale # (NxH)@(HxN) -> (NxN)

        if self.sigma is not None:
        # ajout du prior gaussien
            i = torch.arange(N).unsqueeze(1).to(x.device)
            j = torch.arange(N).unsqueeze(0).to(x.device)
            B_mat = torch.exp(-((j - i) ** 2) / (2 * (self.sigma ** 2))).tril(0)
            score += B_mat.unsqueeze(0)
            
        if hierarchical_mask is not None:
            score += hierarchical_mask.unsqueeze(0) 

        causal = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = score.masked_fill(causal.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn) # (NxN)
        
        return attn @ value # (NxN)@(NxH) -> (NxH)

class MultiHead_Attention(nn.Module):
    """
    Classe gérant le MultiHead Self Attention (dans le papier ils utilisent 4 têtes avec un sigma spécifique par tête (pour le prior gaussien))
    """
    def __init__(
        self,
        n_heads: int,
        N: int,
        d_model: int,
        sigma_list: Optional[List[Optional[float]]] = None,
        pdrop: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"La taille d'entrée ({d_model}) doit être divisible par le nombre de têtes ({n_heads})."
            )
        if sigma_list is not None and len(sigma_list) != n_heads:
            raise ValueError(
                f"La longueur de sigma_list ({len(sigma_list)}) doit correspondre au nombre de têtes ({n_heads})."
            )
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.out_proj_drop = nn.Dropout(pdrop)
        self.heads = nn.ModuleList([
            Head_Attention(d_model, self.head_size, sigma_list[i] if sigma_list else None)
            for i in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, hierarchical_mask=None):
        out = [head(x, hierarchical_mask=hierarchical_mask) for head in self.heads]
        return self.out_proj_drop(self.out_proj(torch.cat(out, dim=-1)))
    
    def _get_penalty(self):
        
        # head.value_linear.weight (head_size x F)
        W = torch.stack([head.value_linear.weight for head in self.heads], dim=0) # (H x head_size x F) 3D
        W = torch.flatten(W, start_dim=1 , end_dim=2) # (H x (head_size*F)) 2D
        W = W / torch.linalg.matrix_norm(W, 2)
        I = torch.eye(W.shape[0], device=W.device)
        penalty = torch.linalg.matrix_norm(W@W.T - I, 'fro')

        return penalty

class SelfAttentionBlock(nn.Module):
    def __init__(self, n_heads:int, N:int, d_model:int,
                 p_drop:float = 0.1,
                 sigma_list:Optional[List[Optional[float]]] = None,
                 hierarchical_mask:Optional[torch.Tensor]=None):
        
        super().__init__()
        
        self.ln1  = nn.LayerNorm(d_model)
        self.mha  = MultiHead_Attention(n_heads, N, d_model, sigma_list)   

        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(                       
            nn.Linear(d_model, 4*d_model),
            nn.Linear(4*d_model, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        self.do2  = nn.Dropout(p_drop)            
        self.hierarchical_mask = hierarchical_mask

    def forward(self, x):
        x = x + self.mha(self.ln1(x), hierarchical_mask=self.hierarchical_mask)
        x = x + self.ff(self.ln2(x))
        return x
    
class TemporalAttention(nn.Module):
    def __init__(self, d_model:int, d_att:int=None,
                 pdrop:float=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.temp_attn_drop = nn.Dropout(pdrop)
        
        d_att = d_att or d_model//2
        
        self.linear = nn.Linear(d_model, d_att, bias=False)
        self.v       = nn.Linear(d_att , 1, bias=False)
        self.out     = nn.Linear(d_model, 1)
        self.act = nn.GELU()

    def forward(self, Z):           
        s = self.ln1(Z)# (B,N,d)
        s = self.linear(s)
        s = self.act(s)
        s = self.v(s).squeeze(-1) # (B,N)  # stabilise
        α = torch.softmax(s, 1).unsqueeze(-1)           
        M = (α*Z).sum(1)
        return self.out(M).squeeze(-1)
    
    