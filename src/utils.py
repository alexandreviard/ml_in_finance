import torch

def get_hierarchical_mask(N: int, horizon: int):
    mask = torch.full((N, N), float('-inf'))
    start = 0
    while start < N:
        end = min(start + horizon, N)
        mask[start:end, start:end] = 0
        start = end
    return mask