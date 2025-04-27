from src.squelette_models import Head_Attention, MultiHead_Attention, SelfAttentionBlock, LearnablePositionalEmbedding, TemporalAttention, SinusoidalPositionalEncoding
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
        self.positional_embedding = SinusoidalPositionalEncoding(N, d_model)
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
        x = self.positional_embedding(torch.tanh(self.linear_input(x)))
        #x = torch.tanh((self.linear_input(self.positional_embedding(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.softmax(self.linear_logits(x))
        
        return x
    

class MG_Daily(nn.Module):
    
    def __init__(self,
                 N=130,
                 F=16,
                 d_model=64,
                 n_heads=4,
                 sigma_list=None,
                 pdrop:float=0.1):
        
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(F)
        self.in_proj  = nn.Linear(F, d_model)
        
        self.in_drop  = nn.Dropout(pdrop*0.5)          
        self.act      = nn.GELU()
        self.pos_emb  = SinusoidalPositionalEncoding(N, F)
        #self.pos_emb  = SinusoidalPositionalEncoding(N, d_model)

        self.block1 = SelfAttentionBlock(n_heads, N, d_model, pdrop, sigma_list)
        self.block2 = SelfAttentionBlock(n_heads, N, d_model, pdrop, sigma_list)
        self.block3 = SelfAttentionBlock(n_heads, N, d_model, pdrop, sigma_list)

        self.temporal_att = TemporalAttention(d_model, pdrop=pdrop)

    def forward(self, x):                                 # (B,N,F)
        x = self.pre_norm(x)
        #x = self.pos_emb(self.act(self.in_proj(x)))
        x = torch.tanh(self.in_proj(self.pos_emb(x)))
        x = self.in_drop(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.temporal_att(x)                       # (B,)
    
    
class CNN_Daily(nn.Module):
    """
    CNN 1‑D (entrée B×N×F  →  sortie B×1).

    • Les convolutions se font sur l’axe temporel N.
    • F = nombre de features est vu comme le nombre de canaux d’entrée.
    """
    def __init__(
        self,
        N: int,          
        F: int,       
        n_filters: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv_block = nn.Sequential(

            nn.Conv1d(in_channels=F, out_channels=n_filters,
                      kernel_size=kernel_size, padding="same"),
            
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=n_filters, out_channels=2*n_filters,
                      kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(2*n_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Linear(2*n_filters, 1)  # logit unique

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, F)  ->  logit : (B,)
        """
        x = x.permute(0, 2, 1)          # (B, F, N) pour Conv1d
        x = self.conv_block(x)          # (B, 2*n_filters, N)
        x = self.global_pool(x).squeeze(-1)  # (B, 2*n_filters)
        return self.fc(x).squeeze(-1)   # (B,)



class LSTM_Daily(nn.Module):
    """
    LSTM 
      entrée  (B, N, F)  ->  logit (B,).
    • batch_first=True pour garder le format (B,N,F) partout
    • On récupère le dernier état caché de la dernière couche,
      puis un FC -> 1 logit.
    """
    
    def __init__(
        self,
        F: int,            
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = F,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
            bidirectional= False
        )

        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, F)  ->  logit : (B,)
        """
        _, (h_n, _) = self.lstm(x)           # h_n : (n_layers, B, hidden)
        h_last = h_n[-1]                     # (B, hidden)
        h_last = self.do(h_last)
        return self.fc(h_last).squeeze(-1)   # (B,)
