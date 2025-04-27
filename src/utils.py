import torch
import requests, warnings
from io import StringIO
import pandas as pd
import torch.nn as nn

def get_hierarchical_mask(N: int, horizon: int):
    mask = torch.full((N, N), float('-inf'))
    start = 0
    while start < N:
        end = min(start + horizon, N)
        mask[start:end, start:end] = 0
        start = end
    return mask


def get_nasdaq_tickers() -> list[str]:
    
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = StringIO("\n".join(resp.text.splitlines()[:-1]))
    df = pd.read_csv(data, sep="|")

    return df["Symbol"].tolist()


def get_nasdaq_100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url, match="Ticker")
    df = tables[0]
    return df["Ticker"].tolist()


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, match="Symbol")
    df = tables[0]
    return df["Symbol"].tolist()


def debug_pass(model, xb):
    hooks = []
    def reg(name):
        def f(_, __, out):
            print(f"{name:15s} μ={out.mean():+.3e} σ={out.std():+.3e}")
        return f

    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.ReLU, nn.LayerNorm)):
            hooks.append(m.register_forward_hook(reg(n)))

    with torch.no_grad():
        _ = model(xb[:2])

    for h in hooks: h.remove()
    


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
        
def print_num_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")