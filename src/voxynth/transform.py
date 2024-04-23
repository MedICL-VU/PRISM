import numpy as np
import torch

from typing import List
from torch import Tensor

from .utility import chance
from .utility import grid_coordinates
from .noise import perlin


def resize(
    image : Tensor,
    scale_factor : float or List[float] = None,
    shape : List[int] = None,
    nearest : bool = False) -> Tensor:
    """
    Resize an image with the option of scaling and/or setting to a new shape.

    Parameters:
    -----------
    image: torch.Tensor
        An input tensor with shape (C, H, W[, D]) to resize.
    scale_factor: float or List[float], optional
        Multiplicative factor(s) for scaling the input tensor. If a float, then the same
        scale factor is applied to all spatial dimensions. If a tuple, then the scaling
        factor for each dimension should be provided.
    shape: List[int], optional
        Target shape of the output tensor.
    nearest: bool, optional
        If True, use nearest neighbor interpolation. Otherwise, use linear interpolation.

    Returns:
    --------
    torch.Tensor:
        The resized tensor with the shape specified by `shape` or scaled by `scale_factor`.
    """
    ndim = image.ndim - 1

    # scale the image if the scale factor is provided
    if scale_factor is not None and scale_factor != 1:

        # compute target shape based on the scale factor
        target_shape = [int(s * scale_factor + 0.5) for s in image.shape[1:]]

        # convert image to float32 if it's not already to enable interpolation
        # if using nearest interpolation, save the original dtype to convert back later
        reset_type = None
        if not torch.is_floating_point(image):
            if nearest:
                reset_type = image.dtype
            image = image.type(torch.float32)

        # determine interpolation mode based on ndim and interpolation type
        linear = 'trilinear' if image.ndim - 1 == 3 else 'bilinear'
        mode = 'nearest' if nearest else linear

        # apply interpolation to the image
        if nearest:
            image = torch.nn.functional.interpolate(image.unsqueeze(0), target_shape, mode=mode)
        else:
            image = torch.nn.functional.interpolate(image.unsqueeze(0), target_shape, mode=mode)
        image = image.squeeze(0)

        # convert image back to its original dtype if necessary
        if reset_type is not None:
            image = image.type(reset_type)

    if shape is not None:

        # compute padding for each spatial dimension
        padding = []
        baseshape = image.shape[1:]
        for d in range(ndim):
            diff = shape[d] - baseshape[d]
            if diff > 0:
                half = diff / 2
                a, b = int(np.floor(half)), int(np.ceil(half))
                padding.extend([a, b])
            else:
                padding.extend([0, 0])

        # apply padding to the image
        padding.reverse()
        image = torch.nn.functional.pad(image, padding)

        # compute slice to remove excess dimensions
        slicing = [slice(0, image.shape[0])]
        baseshape = image.shape[1:]
        for d in range(ndim):
            diff = baseshape[d] - shape[d]
            if diff > 0:
                half = diff / 2
                a, b = int(np.floor(half)), int(np.ceil(half))
                slicing.append(slice(a, baseshape[d] - b))
            else:
                slicing.append(slice(0, baseshape[d]))

        # apply slice to remove excess dimensions
        image = image[tuple(slicing)]

    return image


def random_flip(dim : int, *args, prob : float = 0.5):
    """
    Randomly flips an image (or set of images) along the given dimension.

    Parameters
    ----------
    dim : int
        The dimension along which to flip. Note that the first dimension
        is the channel dimension.
    *args : torch.Tensor
        The image(s) to flip.
    prob : float
        The probability of flipping the image(s).

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor]
        The flipped image(s).
    """
    result = tuple([arg.flip([dim]) for arg in args]) if chance(prob) else args
    if len(args) == 1:
        return result[0]
    return result


def compose_affine(
    ndim : int,
    translation : Tensor = None,
    rotation : Tensor = None,
    scale : Tensor = None,
    shear : Tensor = None,
    degrees : bool = True,
    device : torch.device = None) -> Tensor:
    """
    Composes an affine matrix from a set of translation, rotation, scale,
    and shear transform components.

    Parameters
    ----------
    ndim (int):
        The number of dimensions of the affine matrix. Must be 2 or 3.
    translation : Tensor, optional
        The translation vector. Must be a vector of size `ndim`. 
    rotation : Tensor, optional
        The rotation angles. Must be a scalar value for 2D affine matrices, 
        and a tensor of size 3 for 3D affine matrices.
    scale : Tensor, optional
        The scaling factor. Can be scalar or vector of size `ndim`.
    shear : Tensor, optional
        The shearing factor. Must be a scalar value for 2D affine matrices, 
        and a tensor of size 3 for 3D affine matrices.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.
    device : torch.device, optional
        The device of the returned matrix.

    Returns
    -------
    Tensor
        The composed affine matrix, as a tensor of shape `(ndim + 1, ndim + 1)`.
    """
    if ndim not in (2, 3):
        raise ValueError(f'affine transform must be 2D or 3D, got ndim {ndim}')

    # check translation
    translation = torch.zeros(ndim) if translation is None else torch.as_tensor(translation)
    if len(translation) != ndim:
        raise ValueError(f'translation must be of shape ({ndim},)')

    # check rotation angles
    expected = 3 if ndim == 3 else 1
    rotation = torch.zeros(expected) if rotation is None else torch.as_tensor(rotation)
    if rotation.ndim == 0 and ndim == 3 or rotation.ndim != 0 and rotation.shape[0] != expected:
        raise ValueError(f'rotation must be of shape ({expected},)')

    # check scaling factor
    scale = torch.ones(ndim) if scale is None else torch.as_tensor(scale)
    if scale.ndim == 0:
        scale = scale.repeat(ndim)
    if scale.shape[0] != ndim:
        raise ValueError(f'scale must be of size {ndim}')

    # check shearing
    expected = 3 if ndim == 3 else 1
    shear = torch.zeros(expected) if shear is None else torch.as_tensor(shear)
    if shear.ndim == 0:
        shear = shear.view(1)
    if shear.shape[0] != expected:
        raise ValueError(f'shear must be of shape ({expected},)')

    # start from translation
    T = torch.eye(ndim + 1, dtype=torch.float64)
    T[:ndim, -1] = translation

    # rotation matrix
    R = torch.eye(ndim + 1, dtype=torch.float64)
    R[:ndim, :ndim] = angles_to_rotation_matrix(rotation, degrees=degrees)

    # scaling
    Z = torch.diag(torch.cat([scale, torch.ones(1, dtype=torch.float64)]))

    # shear matrix
    S = torch.eye(ndim + 1, dtype=torch.float64)
    S[0][1] = shear[0]
    if ndim == 3:
        S[0][2] = shear[1]
        S[1][2] = shear[2]

    # compose component matrices
    matrix = T @ R @ Z @ S

    return torch.as_tensor(matrix, dtype=torch.float32, device=device)


def angles_to_rotation_matrix(
    rotation : Tensor,
    degrees : bool = True) -> Tensor:
    """
    Compute a rotation matrix from the given rotation angles.

    Parameters
    ----------
    rotation : Tensor
        A tensor containing the rotation angles. If `degrees` is True, the angles
        are in degrees, otherwise they are in radians.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.

    Returns
    -------
    Tensor
        The computed `(ndim + 1, ndim + 1)` rotation matrix.
    """
    if degrees:
        rotation = torch.deg2rad(rotation)

    # scalar value allowed for 2D transforms
    rotation = torch.as_tensor(rotation)
    if rotation.ndim == 0:
        rotation = rotation.view(1)
    num_angles = len(rotation)

    # build the matrix
    if num_angles == 1:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    elif num_angles == 3:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        rx = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[1]), torch.sin(rotation[1])
        ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[2]), torch.sin(rotation[2])
        rz = torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
        matrix = rx @ ry @ rz
    else:
        raise ValueError(f'expected 1 (2D) or 3 (3D) rotation angles, got {num_angles}')

    return matrix.to(rotation.device)


def random_affine(
    ndim: int,
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 1,
    device: torch.device = None) -> Tensor:
    """
    TODOC
    """

    # 
    translation_range = sorted([-max_translation, max_translation])
    translation = np.random.uniform(*translation_range, size=ndim)

    # 
    rotation_range = sorted([-max_rotation, max_rotation])
    rotation = np.random.uniform(*rotation_range, size=(1 if ndim == 2 else 3))

    # 
    if max_scaling < 1:
        raise ValueError('max scaling to random affine cannot be less than 1, '
                         'see function doc for more info')
    inv = np.random.choice([-1, 1], size=ndim)
    scale = np.random.uniform(1, max_scaling, size=ndim) ** inv

    # compose from random paramters
    aff = compose_affine(
        ndim=ndim,
        translation=translation,
        rotation=rotation,
        scale=scale,
        device=device)
    return aff


def affine_to_displacement_field(
    affine : Tensor,
    meshgrid : Tensor,
    rotate_around_center : bool = True) -> Tensor:
    """
    Convert an affine transformation matrix to a dense displacement field.

    Parameters
    ----------
    affine : Tensor
        Affine transformation matrix.
    meshgrid : Tensor
        The meshgrid tensor of shape `(W, H[, D], N)`, where N is the spatial dimensionality.
    rotate_around_center : bool, optional
        If True, the rotation will be around the center of the image, otherwise around the origin.

    Returns
    -------
    Tensor
        The generated displacement field of shape `meshgrid.shape[:-1]`.
    """
    ndim = meshgrid.shape[-1]
    shape = meshgrid.shape[:-1]

    # if rotate_around_center is enabled, adjust the meshgrid so that the rotation
    # is around the center of the image instead of the origin
    grid = meshgrid.clone() if rotate_around_center else meshgrid
    if rotate_around_center:
        for d in range(ndim):
            grid[..., d] -= (shape[d] - 1) / 2

    # convert the meshgrid to homogeneous coordinates by appending a column of ones
    coords = grid.view(-1, ndim)
    ones = torch.ones((coords.shape[-2], 1), device=meshgrid.device)
    coords = torch.cat([coords, ones], dim=-1)

    # apply the affine transformation to the coordinates to get the shift vector
    shift = (affine @ coords.T)[:ndim].T

    # reshape the shift vector to match the shape of the meshgrid and subtract
    # the original meshgrid to get the displacement field
    shift = shift.view(*shape, ndim) - grid

    return shift


def integrate_displacement_field(
    disp : Tensor,
    steps : int,
    meshgrid : Tensor = None) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        meshgrid = grid_coordinates(disp.shape[:-1], device=disp.device)

    if steps == 0:
        return disp

    disp = disp / (2 ** steps)
    for _ in range(steps):
        disp += spatial_transform(disp.movedim(-1, 0), disp, meshgrid=meshgrid).movedim(0, -1)

    return disp


def random_displacement_field(
    shape : List[int],
    smoothing : float = 10,
    magnitude : float = 10,
    integrations : int = 0,
    voxsize : float = 1,
    meshgrid : Tensor = None,
    device: torch.device = None) -> Tensor:
    """
    TODOC
    """
    smoothing = smoothing / voxsize
    magnitude = magnitude / voxsize

    ndim = len(shape)
    disp = [perlin(shape, smoothing, magnitude, device=device) for i in range(ndim)]
    disp = torch.stack(disp, dim=-1)

    if integrations > 0:
        disp = integrate_displacement_field(disp, integrations, meshgrid)

    return disp


def displacement_field_to_coords(disp, meshgrid=None) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        meshgrid = grid_coordinates(disp.shape[:-1], device=disp.device)

    shape = disp.shape[:-1]
    ndim = disp.shape[-1]

    coords = (meshgrid + disp)
    for d in range(ndim):
        coords[..., d] *= 2 / (shape[d] - 1)
        coords[..., d] -= 1

    coords = coords.flip(-1)

    return coords


def coords_to_displacement_field(coords, meshgrid=None) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        meshgrid = grid_coordinates(coords.shape[:-1], device=coords.device)

    raise NotImplementedError('coords_to_displacement_field is not yet implemented. '
                              'contact andrew if you get this... or implement it :)')


def spatial_transform(
    image : Tensor,
    trf : Tensor,
    method : str = 'linear',
    isdisp : bool = True,
    meshgrid : Tensor = None,
    rotate_around_center : bool = True) -> Tensor:
    """
    TODOC
    """
    if trf is None:
        return image

    if trf.ndim == 2:
        if meshgrid is None:
            meshgrid = grid_coordinates(image.shape[1:], device=image.device)
        trf = torch.linalg.inv(trf)
        trf = affine_to_displacement_field(trf, meshgrid,
                    rotate_around_center=rotate_around_center)
        isdisp = True

    # 
    if isdisp:
        trf = displacement_field_to_coords(trf, meshgrid=meshgrid)

    # 
    method = 'bilinear' if method == 'linear' else method

    # 
    reset_type = None
    if not torch.is_floating_point(image):
        if method == 'nearest':
            reset_type = image.dtype
        image = image.type(torch.float32)

    # 
    image = image.unsqueeze(0)
    trf = trf.unsqueeze(0)
    interped = torch.nn.functional.grid_sample(image, trf, align_corners=True, mode=method)
    interped = interped.squeeze(0)

    # 
    if reset_type is not None:
        interped = interped.type(reset_type)

    return interped


def random_transform(
    shape : List[int],
    affine_probability : float = 1.0,
    max_translation : float = 5.0,
    max_rotation : float = 5.0,
    max_scaling : float = 1.1,
    warp_probability : float = 1.0,
    warp_integrations : int = 5,
    warp_smoothing_range : List[int] = [10, 20],
    warp_magnitude_range : List[int] = [1, 2],
    voxsize : int = 1,
    device : torch.device = None,
    isdisp : bool = True,
    ) -> Tensor:
    """
    TODOC
    """
    ndim = len(shape)
    trf = None

    # generate a random affine
    if chance(affine_probability):

        # compute meshgrid
        meshgrid = grid_coordinates(shape, device=device)

        # scale max translation value so that it correctly corresponds to mm
        max_translation = max_translation / voxsize
        matrix = random_affine(
            ndim=ndim,
            max_translation=max_translation,
            max_rotation=max_rotation,
            max_scaling=max_scaling,
            device=device)
        trf = affine_to_displacement_field(matrix, meshgrid)

    # generate a nonlinear transform
    if chance(warp_probability):
        disp = random_displacement_field(
            shape=shape,
            smoothing=np.random.uniform(*warp_smoothing_range),
            magnitude=np.random.uniform(*warp_magnitude_range),
            integrations=warp_integrations,
            voxsize=voxsize,
            device=device)

        # merge with the affine transform if necessary
        if trf is None:
            trf = disp
        else:
            trf += spatial_transform(disp.movedim(-1, 0), trf, meshgrid=meshgrid).movedim(0, -1)

    # convert to coordinates if specified
    if trf is not None and not isdisp:
        trf = displacement_field_to_coords(trf)

    return trf
