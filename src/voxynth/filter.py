import torch

from typing import List
from torch import Tensor


def gaussian_kernel(
    sigma: List[float],
    truncate: int = 3,
    device: torch.device = None) -> Tensor:
    """
    Generate a Gaussian kernel with the specified standard deviations.

    Parameters
    ----------
    sigma : List[float]
        A list of standard deviations for each dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.
    device : torch.device, optional
        The device on which to create the kernel.

    Returns
    -------
    Tensor
        A kernel of shape `(2 * truncate * sigma + 1,) * ndim`.

    Notes
    -----
    The kernel is truncated when its values drop below `1e-5` of the maximum value.
    """
    ndim = len(sigma)

    # compute the radii of the kernel for each dimension
    radii = [int(truncate * s + 0.5) for s in sigma]

    # generate a range of indices for each dimension
    ranges = [torch.arange(-r, r + 1, device=device) for r in radii]

    # create a meshgrid of indices for all dimensions and determine shape of the kernel
    coords = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    kernel_shape = coords.shape[:-1]

    # convert the standard deviations to a tensor and compute the inverse squares
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    sigma2 = 1 / torch.clip(sigma, min=1e-5).pow(2)

    # reshape the coordinates and compute the pdf
    coords = coords.view(-1, ndim)
    pdf = torch.exp(-0.5 * (coords.pow(2) * sigma2).sum(-1)).view(kernel_shape)

    # normalize the kernel
    pdf /= pdf.sum()
    return pdf


def gaussian_kernel_1d(sigma, truncate: int = 3, device=None, dtype=None):
    """
    Generate a 1D Gaussian kernel with the specified standard deviations.

    Parameters
    ----------
    sigma : float
        A list of standard deviations for each dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.
    device : torch.device, optional
        The device on which to create the kernel.
    dtype : torch.dtype | None, optional
        Data type of the returned kernel.

    Returns
    -------
    Tensor
        A kernel of shape `2 * truncate * sigma + 1`.

    Notes
    -----
    The kernel is truncated when its values drop below `1e-5` of the maximum value.
    """
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    sigma2 = 1 / torch.clip(torch.as_tensor(sigma), min=1e-5).pow(2)
    pdf = torch.exp(-0.5 * (x.pow(2) * sigma2))
    return pdf / pdf.sum()


def gaussian_blur(
    image: Tensor,
    sigma: List[float],
    batched: bool = False,
    truncate: int = 3) -> Tensor:
    """
    Apply Gaussian blurring to an image.

    Parameters
    ----------
    image : Tensor
        An input tensor of shape `(C, W, H[, D])` to blur. A batch dimension
        can be included by setting `batched` to `True`.
    sigma : float or List[float]
        Standard deviation(s) of the Gaussian filter along each dimension.
    batched : bool, optional
        Whether the input tensor includes a batch dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.

    Returns
    -------
    Tensor
        The blurred tensor with the same shape as the input tensor.

    Notes
    -----
    The Gaussian filter is applied using convolution. The size of the filter kernel is
    determined by the standard deviation and the truncation factor.
    """
    ndim = image.ndim - (2 if batched else 1)

    # sanity check for common mistake
    if ndim == 4 and not batched:
        raise ValueError(f'gaussian blur input has {image.ndim} dims, '
                          'but batched option is False')

    # normalize sigmas
    if torch.as_tensor(sigma).ndim == 0:
        sigma = [sigma] * ndim
    if len(sigma) != ndim:
        raise ValueError(f'sigma must be {ndim}D, but got length {len(sigma)}')

    blurred = image if batched else image.unsqueeze(0)

    if all(s == sigma[0] for s in sigma):
        # Isotropic, can use the same vector in all directions cases. Since
        # creating the kernel is actually one of the most time intensive steps
        # this is an efficiency gain worth exploiting
        kernel_vec = gaussian_kernel_1d(
            sigma[0],
            truncate,
            device=blurred.device,
            dtype=blurred.dtype,
        )
        kernel_vecs = [kernel_vec] * ndim
    else:
        # Three different kernels, one for each direction
        kernel_vecs = [
            gaussian_kernel_1d(
                s,
                truncate,
                device=blurred.device,
                dtype=blurred.dtype,
            )
            for s in sigma
        ]

    for dim, kernel in enumerate(kernel_vecs):

        # apply the convolution
        slices = [None] * (ndim + 2)
        slices[dim + 2] = slice(None)
        kernel_dim = kernel[slices]
        conv = getattr(torch.nn.functional, f'conv{ndim}d')
        blurred = conv(blurred, kernel_dim, groups=image.shape[0], padding="same")

    if not batched:
        blurred = blurred.squeeze(0)

    return blurred
