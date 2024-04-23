import math
import torch

from typing import List
from torch import Tensor

from .filter import gaussian_blur


def perlin(shape, smoothing, magnitude=1.0, device=None):
    """
    Generates a perlin noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    smoothing : float or List[float]
        The spatial smoothing sigma in voxel coordinates. If a
        single value is provided, it will be used for all dimensions.
    magnitude : float or List[float]
        The standard deviation of the noise across dimensions. If a single value is
        provided, it will be used for all dimensions.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.

    Returns
    -------
    Tensor
        A Perlin noise image of shape `shape`.
    """
    noise = torch.normal(0, 1, size=shape, device=device).unsqueeze(0)
    noise = gaussian_blur(noise, smoothing).squeeze(0)
    noise *= magnitude / noise.std()
    return noise
