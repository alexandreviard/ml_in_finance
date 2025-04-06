import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np

class LearnablePositionalEmbedding(nn.Module):
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
    d_model : la dimension du modèle après la projection de (NxF) vers (Nxd_model)
    """
    def __init__(self, d_model: int, head_size: int, sigma: Optional[float] = None):
        super().__init__()
        self.head_size = head_size
        self.key_linear = nn.Linear(d_model, head_size)
        self.query_linear = nn.Linear(d_model, head_size)
        self.value_linear = nn.Linear(d_model, head_size)
        self.scale = torch.tensor(1 / torch.sqrt(torch.tensor(d_model)))
        self.sigma = sigma

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
        d_model: int,
        sigma_list: Optional[List[Optional[float]]] = None
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
        self.heads = nn.ModuleList([
            Head_Attention(d_model, self.head_size, sigma_list[i] if sigma_list else None)
            for i in range(n_heads)
        ])

    def forward(self, x, hierarchical_mask=None):
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
        d_model: int,
        sigma_list: Optional[List[Optional[float]]] = None,
        hierarchical_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) # normalise au niveau de F, puis rescale chaque feature de F avec 2 paramètres appris par feature
        self.mha = MultiHead_Attention(n_heads, N, d_model, sigma_list)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.hierarchical_mask = hierarchical_mask

    def forward(self, x):
        x = x + self.mha(self.ln1(x), hierarchical_mask=self.hierarchical_mask)
        x = x + self.ff(self.ln2(x))
        return x
    
class TemporalAttention(nn.Module):
    """
    Implémente une attention temporelle similaire à Qin et al. (2017).
    e_i = v^T tanh(W_alpha * z_i + b_alpha)
    alpha_i = softmax(e_i)
    """
    def __init__(self, d_model, d_att):
        """
        d_model : dimension du vecteur z_i (sortie du transformer)
        d_att   : dimension cachée pour le calcul de l'attention
        """
        super(TemporalAttention, self).__init__()
        self.W_alpha = nn.Linear(d_model, d_att, bias=True)
        self.v = nn.Linear(d_att, 1, bias=False)

    def forward(self, Z):
        """
        Z : [batch_size, seq_len, d_model]
            => Z[:, i, :] = z_i
        
        Retourne:
          alpha : [batch_size, seq_len]
          M     : [batch_size, d_model] (somme pondérée)
        """
        # 1) Projection + tanh
        Wh = self.W_alpha(Z)          # [batch_size, seq_len, d_att]
        Wh_tanh = torch.tanh(Wh)      # [batch_size, seq_len, d_att]
        
        # 2) Scores e_i
        e = self.v(Wh_tanh)           # [batch_size, seq_len, 1]
        e = e.squeeze(-1)             # [batch_size, seq_len]
        
        # 3) alpha = softmax(e_i)
        alpha = F.softmax(e, dim=1)   # [batch_size, seq_len]
        
        # 4) Agrégation M = sum_i alpha_i * z_i
        alpha_3d = alpha.unsqueeze(-1)        # [batch_size, seq_len, 1]
        M = (alpha_3d * Z).sum(dim=1)         # [batch_size, d_model]
        
        return alpha, M

class TemporalAttention(nn.Module):

    def __init__(self, d_model, d_att):
        super(TemporalAttention, self).__init__()
        self.W_alpha = nn.Linear(d_model, d_att, bias=True)
        self.v = nn.Linear(d_att, 1, bias=False)
        self.Wfc = nn.Linear(d_model, 1)

    def forward(self, Z):

        Wh = self.W_alpha(Z)          # [batch_size, seq_len, d_att]
        Wh_tanh = torch.tanh(Wh)      # [batch_size, seq_len, d_att]
        
        e = self.v(Wh_tanh)           # [batch_size, seq_len, 1]
        e = e.squeeze(-1)             # [batch_size, seq_len]
        
        alpha = torch.softmax(e, dim=1)   # [batch_size, seq_len]
        alpha_3d = alpha.unsqueeze(-1)        # [batch_size, seq_len, 1]
        M = (alpha_3d * Z).sum(dim=1)       # [batch_size, d_model]
        
        M = torch.sigmoid(self.Wfc(M)) # [batch_size, 1]
        
        return M