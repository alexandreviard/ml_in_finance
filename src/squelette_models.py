import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np

class LearnablePositionalEmbedding(nn.Module):
    """
    Classe qui gère l'injection d'embedding positionnel pour chaque point temporel.
    """
    def __init__(self, N: int, F: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, N, F))

    def forward(self, x):
        x = x + self.pos_embedding
        return x

class Head_Attention(nn.Module):
    """
    Classe qui définit une tête d'attention (self-attention) avec un éventuel prior gaussien.
    N : la taille de la window considéré (en horizon temporel)
    F : la dimension des features pour un actif fixé (pourquoi pas ajouter plusieurs actifs et prédire leur hausse/baisse F = n_actifs*feature_par_actif ?)
    """
    def __init__(self, F: int, head_size: int, sigma: Optional[float] = None):
        super().__init__()
        self.F = F
        self.head_size = head_size
        self.key_linear = nn.Linear(F, head_size)
        self.query_linear = nn.Linear(F, head_size)
        self.value_linear = nn.Linear(F, head_size)
        self.scale = torch.tensor(1 / torch.sqrt(torch.tensor(head_size)))
        self.sigma = sigma

    def forward(self, x, hierarchical_mask=None):
        B, N, F = x.shape
        key = self.key_linear(x) # (NxF) -> (NxH)
        query = self.query_linear(x) # (NxF) -> (NxH)
        value = self.value_linear(x) # (NxF) -> (NxH)
        score = (query @ key.transpose(-2, -1)) * self.scale # (NxH)@(HxN) -> (NxN)

        if self.sigma is not None:
        # ajout du prior gaussien
            i = torch.arange(N).unsqueeze(1).to(x.device)
            j = torch.arange(N).unsqueeze(0).to(x.device)
            B_mat = torch.exp(-((j - i) ** 2) / (2 * (self.sigma ** 2))).tril(0)
            score += B_mat.unsqueeze(0)
            
        if hierarchical_mask:
            score += hierarchical_mask.unsqueeze(0) 

        mask = torch.triu(torch.ones(N, N), diagonal=1).bool().to(x.device)
        score.masked_fill_(mask.unsqueeze(0), float('-inf'))
        return torch.softmax(score, dim=-1) @ value

class MultiHead_Attention(nn.Module):
    """
    Classe gérant le MultiHead Self Attention (dans le papier ils utilisent 4 têtes avec un sigma spécifique par tête (pour le prior gaussien))
    """
    def __init__(
        self,
        n_heads: int,
        N: int,
        F: int,
        sigma_list: Optional[List[Optional[float]]] = None
    ):
        super().__init__()
        if F % n_heads != 0:
            raise ValueError(
                f"La taille d'entrée ({F}) doit être divisible par le nombre de têtes ({n_heads})."
            )
        if sigma_list is not None and len(sigma_list) != n_heads:
            raise ValueError(
                f"La longueur de sigma_list ({len(sigma_list)}) doit correspondre au nombre de têtes ({n_heads})."
            )
        self.n_heads = n_heads
        self.head_size = F // n_heads
        self.positional_embedding = LearnablePositionalEmbedding(N, F)
        self.heads = nn.ModuleList([
            Head_Attention(F, self.head_size, sigma_list[i] if sigma_list else None)
            for i in range(n_heads)
        ])

    def forward(self, x, hierarchical_mask=None):
        x = self.positional_embedding(x)
        out = [head(x, hierarchical_mask=hierarchical_mask) for head in self.heads]
        return torch.cat(out, dim=-1)
    
    def _get_penalty(self):
        
        # head.value_linear.weight (head_size x F)
        W = torch.stack([head.value_linear.weight for head in self.heads], dim=0) # (H x head_size x F) 3D
        W = torch.flatten(W, start_dim=1 , end_dim=2) # (H x (head_size*F)) 2D
        W = W / torch.linalg.matrix_norm(W, 2)
        I = torch.eye(W.shape[0], device=W.device)
        penalty = torch.linalg.matrix_norm(W@W.T - I, 'fro')

        return penalty

class SelfAttentionBlock(nn.Module):
    """
    Classe qui définit le bloc de self-attention complet comprenant :
    - Normalisation par couche
    - Multi-Head Attention
    - Connexion résiduel
    - Normalisation par couche
    - Feed-Forward (MLP) de taille 4 fois supérieure
    - Connexion Residuel
    """
    def __init__(
        self,
        n_heads: int,
        N: int,
        F: int,
        sigma_list: Optional[List[Optional[float]]] = None,
        hierarchical_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(F) # normalise au niveau de F, puis rescale chaque feature de F avec 2 paramètres appris par feature
        self.mha = MultiHead_Attention(n_heads, N, F, sigma_list)
        self.ln2 = nn.LayerNorm(F)
        self.ff = nn.Sequential(
            nn.Linear(F, 4 * F),
            nn.GELU(),
            nn.Linear(4 * F, F)
        )
        
        self.hierarchical_mask = hierarchical_mask

    def forward(self, x):
        x = x + self.mha(self.ln1(x), hierarchical_mask=self.hierarchical_mask)
        x = x + self.ff(self.ln2(x))
        return x