import numpy as np
import torch
from torch import Tensor


def seed(seed: int = None):
    """
    Sets the seed for both the numpy and PyTorch random number generators.

    Parameters
    ----------
    seed : int or None, optional
        Seed value to be used for random number generation. If None (default),
        the seed is set to 0.
    """
    if seed is None:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


def chance(prob: float) -> bool:
    """
    Returns True with given probability.

    Parameters
    ----------
    prob : float
        Probability of returning True. Must be in the range [0, 1].

    Returns
    -------
    bool
        True with probability `prob`.
    """
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f'chance() expected a value in the range [0, 1], but got {prob}')
    return np.random.rand() < prob


def grid_coordinates(shape, device : torch.device = None) -> Tensor:
    """
    TODOC
    """
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in shape]
    meshgrid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    return meshgrid


def quantile(arr, q):
    """
    TODOC
    """
    if q < 0 or q > 1:
        raise ValueError(f'quantile must be between 0 and 1, got {q}')
    if q == 0:
        return arr.min()
    if q == 1:
        return arr.max()
    arr = arr.flatten()
    if q > 0.5:
        k = int(arr.numel() * (1.0 - q)) + 1
        return arr.topk(k, largest=True, sorted=False).values.min()
    else:
        k = int(arr.numel() * q) + 1
        return arr.topk(k, largest=False, sorted=False).values.max()
