"""3D structure tensor module using CuPy."""

import logging
from typing import Literal

import cupy as lib
import cupy.typing as libt
import cupyx as cpx
from cupyx.scipy import ndimage


def structure_tensor_3d(
    volume: libt.ArrayLike,
    sigma: float,
    rho: float,
    out: libt.NDArray | None = None,
    truncate: float = 4.0,
) -> libt.NDArray:
    """Calculate the structure tensor for 3D image data.

    Args:
        volume: A 3D array. Pass `cupy.ndarray` to avoid copying volume.
        sigma: A noise scale, structures smaller than sigma will be removed by smoothing.
        rho: An integration scale giving the size over the neighborhood in which the orientation is to be analysed.
        out: An array with the shape `(6, ...)` in which to place the output. If `None`, a new array is created.
        truncate: Truncate the filter at this many standard deviations.
    Returns:
        S: An array with shape `(6, ...)` containing elements of structure tensor `(s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)`.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2024
    """

    # Make sure it's an array.
    volume = lib.asarray(volume)

    # Check data type. Must be floating point.
    if not lib.issubdtype(volume.dtype, lib.floating):
        logging.warning(
            "volume is not floating type array. This may result in a loss of precision and unexpected behavior."
        )

    # Computing derivatives (scipy implementation truncates filter at 4 sigma).
    Vx = ndimage.gaussian_filter(volume, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)  # type: ignore
    Vy = ndimage.gaussian_filter(volume, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)  # type: ignore
    Vz = ndimage.gaussian_filter(volume, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)  # type: ignore

    if out is None:
        # Allocate S.
        S = lib.empty((6,) + volume.shape, dtype=volume.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    # Integrating elements of structure tensor (scipy uses sequence of 1D).
    lib.multiply(Vx, Vx, out=S[0])
    ndimage.gaussian_filter(S[0], rho, mode="nearest", output=S[0], truncate=truncate)
    lib.multiply(Vy, Vy, out=S[1])
    ndimage.gaussian_filter(S[1], rho, mode="nearest", output=S[1], truncate=truncate)
    lib.multiply(Vz, Vz, out=S[2])
    ndimage.gaussian_filter(S[2], rho, mode="nearest", output=S[2], truncate=truncate)
    lib.multiply(Vx, Vy, out=S[3])
    ndimage.gaussian_filter(S[3], rho, mode="nearest", output=S[3], truncate=truncate)
    lib.multiply(Vx, Vz, out=S[4])
    ndimage.gaussian_filter(S[4], rho, mode="nearest", output=S[4], truncate=truncate)
    lib.multiply(Vy, Vz, out=S[5])
    ndimage.gaussian_filter(S[5], rho, mode="nearest", output=S[5], truncate=truncate)

    return S


def eig_special_3d(
    S: libt.ArrayLike,
    full: bool = False,
    eigenvalue_order: Literal["desc", "asc"] = "desc",
) -> tuple[libt.NDArray, libt.NDArray]:
    """Eigensolution for symmetric real 3-by-3 matrices.

    Args:
        S: A floating point array with shape (6, ...) containing structure tensor. Use `float64` to avoid numerical errors. When using lower precision, ensure that the values of S are not very small/large. Pass `cupy.ndarray` to avoid copying S.
        full: A flag indicating that all three eigenvectors should be returned.
        eigenvalue_order: The order of eigenvalues. Either "desc" for descending or "asc" for ascending. If all three eigenvectors are returned, they will be ordered according to the eigenvalues.

    Returns:
        val: An array with shape `(3, ...)` containing eigenvalues.
        vec: An array with shape `(3, ...)` containing the vector corresponding to the smallest eigenvalue if `full` is `False`, otherwise `(3, 3, ...)` containing eigenvectors.

    More:
        An analytic solution of eigenvalue problem for real symmetric matrix,
        using an affine transformation and a trigonometric solution of third
        order polynomial. See https://en.wikipedia.org/wiki/Eigenvalue_algorithm
        which refers to Smith's algorithm https://dl.acm.org/citation.cfm?id=366316.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2024
    """
    S = lib.asarray(S)

    # Check data type. Must be floating point.
    if not lib.issubdtype(S.dtype, lib.floating):
        raise ValueError("S must be floating point type.")

    # Flatten S.
    input_shape = S.shape
    S = S.reshape(6, -1)

    # Create v vector.
    v = lib.array([[2 * lib.pi / 3], [4 * lib.pi / 3]], dtype=S.dtype)

    # Computing eigenvalues.

    # Allocate vec and val. We will use them for intermediate computations as well.
    if full:
        val = lib.empty((3,) + S.shape[1:], dtype=S.dtype)
        vec = lib.empty((9,) + S.shape[1:], dtype=S.dtype)
        tmp = lib.empty((4,) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec[:3]
    else:
        val = lib.empty((3,) + S.shape[1:], dtype=S.dtype)
        vec = lib.empty((3,) + S.shape[1:], dtype=S.dtype)
        tmp = lib.empty((4,) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec

    # Views for B.
    B0 = B03[0]
    B1 = B03[1]
    B2 = B03[2]
    B3 = B36[0]
    B4 = B36[1]
    B5 = B36[2]

    # Compute q, mean of diagonal. We need to use q multiple times later.
    # Using lib.mean has precision issues.
    q = lib.add(S[0], S[1], out=tmp[0])
    q += S[2]
    q /= 3

    # Compute S minus q. Insert it directly into B where it'll stay.
    Sq = lib.subtract(S[:3], q, out=B03)

    # Compute s, off-diagonal elements. Store in part of B not yet used.
    s = lib.sum(lib.multiply(S[3:], S[3:], out=B36), axis=0, out=tmp[1])
    s *= 2

    # Compute p.
    p = lib.sum(lib.multiply(Sq, Sq, out=B36), axis=0, out=tmp[2])
    del Sq  # Last use of Sq.
    p += s

    p *= 1 / 6
    lib.sqrt(p, out=p)

    # Compute inverse p, while avoiding 0 division.
    # Reuse s allocation and delete s variable.
    p_inv = s
    del s
    non_zero_p_mask = p == 0
    lib.divide(1, p, out=p_inv)
    p_inv[non_zero_p_mask] = 0

    # Compute B. First part is already filled.
    B03 *= p_inv
    lib.multiply(S[3:], p_inv, out=B36)

    # Compute d, determinant of B.
    d = lib.prod(B03, axis=0, out=tmp[3])

    # Reuse allocation for p_inv and delete variable.
    d_tmp = p_inv
    del p_inv
    # Computation of d.
    lib.multiply(B2, B3, d_tmp)
    d_tmp *= B3
    d -= d_tmp
    lib.multiply(B4, B4, out=d_tmp)
    d_tmp *= B1
    d -= d_tmp
    lib.prod(B36, axis=0, out=d_tmp)
    d_tmp *= 2
    d += d_tmp
    lib.multiply(B5, B5, out=d_tmp)
    d_tmp *= B0
    d -= d_tmp
    d *= 0.5
    # Ensure -1 <= d/2 <= 1.
    lib.clip(d, -1, 1, out=d)

    # Compute phi. Beware that we reuse d variable!
    phi = d
    del d
    phi = lib.arccos(phi, out=phi)
    phi /= 3

    # Compute val, ordered eigenvalues. Resuing B allocation.
    del B03, B36, B0, B1, B2, B3, B4, B5

    lib.add(v, phi[lib.newaxis], out=val[:2])
    val[2] = phi
    lib.cos(val, out=val)
    p *= 2
    val *= p
    val += q

    # Remove all variable using tmp allocation.
    del q
    del p
    del phi
    del d_tmp

    # Computing eigenvectors -- either only one or all three.
    if full:
        l = val
        vec = vec.reshape(3, 3, -1)
        vec_tmp = tmp[:3]
    else:
        l = val[0]
        vec_tmp = tmp[2]

    # Compute vec. The tmp variable can be reused.

    # u = S[4] * S[5] - (S[2] - l) * S[3]
    u = lib.subtract(S[2], l, out=vec[0])
    lib.multiply(u, S[3], out=u)
    u_tmp = lib.multiply(S[4], S[5], out=tmp[3])
    lib.subtract(u_tmp, u, out=u)
    # Put values of u into vector 2 aswell.

    # v = S[3] * S[5] - (S[1] - l) * S[4]
    v = lib.subtract(S[1], l, out=vec_tmp)
    lib.multiply(v, S[4], out=v)
    v_tmp = lib.multiply(S[3], S[5], out=tmp[3])
    lib.subtract(v_tmp, v, out=v)

    # w = S[3] * S[4] - (S[0] - l) * S[5]
    w = lib.subtract(S[0], l, out=vec[2])
    lib.multiply(w, S[5], out=w)
    w_tmp = lib.multiply(S[3], S[4], out=tmp[3])
    lib.subtract(w_tmp, w, out=w)

    vec[1] = u
    lib.multiply(u, v, out=vec[0])
    u = vec[1]
    lib.multiply(u, w, out=vec[1])
    lib.multiply(v, w, out=vec[2])

    # Remove u, v, w and l variables.
    del u
    del v
    del w
    del l

    # Normalizing -- depends on number of vectors.
    if full:
        # vec is [x1 x2 x3, y1 y2 y3, z1 z2 z3]
        l = lib.sum(lib.square(vec), axis=0, out=vec_tmp)[:, lib.newaxis]
        vec = lib.swapaxes(vec, 0, 1)
    else:
        # vec is [x1 y1 z1] = v1
        l = lib.sum(lib.square(vec, out=tmp[:3]), axis=0, out=vec_tmp)

    cpx.rsqrt(l, out=l)
    vec *= l

    val = val.reshape(val.shape[:-1] + input_shape[1:])
    vec = vec.reshape(vec.shape[:-1] + input_shape[1:])

    if eigenvalue_order == "desc":
        val = lib.flip(val, axis=0)
        if full:
            vec = lib.flip(vec, axis=1)

    return val, vec
