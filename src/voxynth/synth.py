import numpy as np
import torch

from typing import Dict
from torch import Tensor

from .utility import chance


def remap_intesities(
    image: Tensor,
    bins: int = 256) -> Tensor:
    """
    Remap image intensities to a random distribution.

    Parameters
    ----------
    image : torch.Tensor
        Image to be remapped
    bins : int, optional
        The number of intensity bins to use when remapping the image.
        The default value of 256 is appropriate for 8-bit images.

    Returns
    -------
    torch.Tensor
        Remapped image
    """
    device = image.device

    image = image.type(torch.float32) - image.min()
    image /= image.max()
    image *= (bins - 1)
    image = image.type(torch.int64)

    # these were somewhat arbitrarily chosen based on
    # testing a few values
    samples = 2
    radians = 1

    # generate noise
    noise = torch.ones(bins, device=device, dtype=torch.float32)
    for i in range(samples):
        low  = np.random.uniform(-radians, 0)
        high = np.random.uniform(0, radians)
        noise *= torch.sin(torch.linspace(low, high, bins,
                    device=device, dtype=torch.float32))

    noise -= noise.min()
    noise /= noise.max()
    return noise[image]


def densities_to_image(densities: Tensor) -> Tensor:
    """
    Replace density values with random signal between 0 and 1.

    Parameters
    ----------
    densities : torch.Tensor
        Multi-channel image of class density (probability) values

    Returns
    -------
    torch.Tensor
        Synthetic image with the same geometry as the input `densities` tensor.
    """
    dims = [1] * (densities.ndim - 1)
    intensities = torch.rand(densities.shape[0], *dims)
    return torch.sum(densities * intensities, axis=0).unsqueeze(0)


def labels_to_image(
    labels: Tensor,
    intensity_ranges: Dict = None) -> Tensor:
    """
    Replace segmentation labels with random signal.

    Parameters
    ----------
    labels : torch.Tensor
        Image of integer segmentation labels
    intensity_ranges : dict, optional
        A dictionary mapping label values to intensity range tuples (low, high).
        Intensity values will be randomly generated within the specified range
        for each label. If not provided, a random intensity will be chosen for
        each label.

    Returns
    -------
    torch.Tensor
        Synthetic image with the same shape as the input `labels` tensor.
    """
    labels = labels.type(torch.int64)
    max_label = labels.max() + 1
    if intensity_ranges is not None:
        mapping = torch.zeros(max_label, device=labels.device, dtype=torch.float32)
        for k, (low, high) in intensity_ranges.items():
            mapping[k] = np.random.uniform(low, high)
    else:
        mapping = torch.rand(max_label, device=labels.device, dtype=torch.float32)
    image = mapping[labels]
    return image
