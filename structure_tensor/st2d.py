"""2D structure tensor module."""

import logging

import numpy as np
import numpy.typing as npt
from scipy import ndimage


def structure_tensor_2d(
    image: npt.ArrayLike,
    sigma: float,
    rho: float,
    out: npt.NDArray | None = None,
    truncate: float = 4.0,
) -> npt.NDArray:
    """Calculate the structure tensor for 2D image data.

    Args:
        volume: A 2D array. Pass `numpy.ndarray` to avoid copying image.
        sigma: A noise scale, structures smaller than sigma will be removed by smoothing.
        rho: An integration scale giving the size over the neighborhood in which the orientation is to be analysed.
        out: An array with the shape `(3, ...)` in which to place the output. If `None`, a new array is created.
        truncate: Truncate the filter at this many standard deviations.
    Returns:
        S: An array with shape `(3, ...)` containing elements of structure tensor `(s_xx, s_yy, s_xy)`.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2024
    """

    # Make sure it's a Numpy array.
    image = np.asarray(image)

    # Check data type. Must be floating point.
    if not np.issubdtype(image.dtype, np.floating):
        logging.warning(
            "image is not floating type array. This may result in a loss of precision and unexpected behavior."
        )

    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Ix = ndimage.gaussian_filter(image, sigma, order=(1, 0), mode="nearest", truncate=truncate)  # type: ignore
    Iy = ndimage.gaussian_filter(image, sigma, order=(0, 1), mode="nearest", truncate=truncate)  # type: ignore

    if out is None:
        # Allocate S.
        S = np.empty((3,) + image.shape, dtype=image.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    # Integrate elements of structure tensor (Scipy uses sequence of 1D).
    tmp = np.empty(image.shape, dtype=image.dtype)
    np.multiply(Ix, Ix, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode="nearest", output=S[0], truncate=truncate)
    np.multiply(Iy, Iy, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode="nearest", output=S[1], truncate=truncate)
    np.multiply(Ix, Iy, out=tmp)
    ndimage.gaussian_filter(tmp, rho, mode="nearest", output=S[2], truncate=truncate)

    return S


def eig_special_2d(S: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """Eigensolution for symmetric real 2-by-2 matrices.

    Args:
        S: A floating point array with shape (3, ...) containing structure tensor. Pass `numpy.ndarray` to avoid copying S.

    Returns:
        val: An array with shape (2, ...) containing sorted eigenvalues.
        vec: An array with shape (2, ...) containing eigenvector corresponding to the smallest eigenvalue (the other is orthogonal to the first).

    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020-2024
    """

    # Save original shape and flatten.
    S = np.asarray(S)

    input_shape = S.shape
    S = S.reshape(3, -1)

    # Calculate val.
    val = np.empty((2, S.shape[1]), dtype=S.dtype)
    np.subtract(S[0], S[1], out=val[1])
    val[1] *= val[1]
    np.multiply(S[2], S[2], out=val[0])
    val[0] *= 4
    val[1] += val[0]
    np.sqrt(val[1], out=val[1])
    np.negative(val[1], out=val[0])
    val += S[0]
    val += S[1]
    val *= 0.5

    # Calcualte vec, y will be positive.
    vec = np.empty((2, S.shape[1]), dtype=S.dtype)
    np.negative(S[2], out=vec[0])
    np.subtract(S[0], val[0], out=vec[1])

    # Deal with diagonal matrices.
    aligned = S[2] == 0

    # Sort.
    vec[:, aligned] = 1 - np.argsort(S[:2, aligned], axis=0)

    # Normalize.
    vec_norm = np.einsum("ij,ij->j", vec, vec)
    np.sqrt(vec_norm, out=vec_norm)
    vec /= vec_norm

    # Reshape and return.
    return val.reshape(val.shape[:1] + input_shape[1:]), vec.reshape(vec.shape[:1] + input_shape[1:])
