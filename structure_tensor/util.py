"""Utilities module."""

from typing import Generator

import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
except Exception as ex:
    cp = None


def _calculate_kernel_radius(sigma, truncate):
    # Multiply by 2 since we do two rounds of filtering when calculating the structure tensor.
    return 2 * int(sigma * truncate + 0.5)


def get_block_count(data: npt.NDArray, block_size: int = 512) -> int:
    """Gets the number of blocks that will be created for the given input."""

    return np.prod(np.ceil(np.array(data.shape) / block_size).astype(int)).item()


def get_block(
    i: int,
    data: npt.NDArray,
    sigma: float,
    block_size: int = 512,
    truncate: float = 4.0,
    copy: bool = False,
) -> tuple[npt.NDArray, np.ndarray, np.ndarray]:
    """Gets the ith block."""

    kernel_radius = _calculate_kernel_radius(sigma, truncate)
    count = 0

    for x0 in range(0, data.shape[0], block_size):
        for y0 in range(0, data.shape[1], block_size):
            for z0 in range(0, data.shape[2], block_size):
                if count == i:
                    x1 = x0 + block_size
                    y1 = y0 + block_size
                    z1 = z0 + block_size

                    block = data[
                        max(0, x0 - kernel_radius) : x1 + kernel_radius,
                        max(0, y0 - kernel_radius) : y1 + kernel_radius,
                        max(0, z0 - kernel_radius) : z1 + kernel_radius,
                    ]

                    cx0 = kernel_radius + min(0, x0 - kernel_radius)
                    cy0 = kernel_radius + min(0, y0 - kernel_radius)
                    cz0 = kernel_radius + min(0, z0 - kernel_radius)
                    cx1 = max(0, min(kernel_radius, data.shape[0] - x1))
                    cy1 = max(0, min(kernel_radius, data.shape[1] - y1))
                    cz1 = max(0, min(kernel_radius, data.shape[2] - z1))

                    if copy:
                        block = np.array(block)

                    return (
                        block,
                        np.array(((x0, x1), (y0, y1), (z0, z1))),
                        np.array(((cx0, cx1), (cy0, cy1), (cz0, cz1))),
                    )

                count += 1

    raise IndexError(f"Index {i} is out of bounds for {count} blocks.")


def get_block_generator(
    data: npt.NDArray,
    sigma: float,
    block_size: int = 512,
    truncate: float = 4.0,
    copy: bool = False,
) -> Generator[tuple[npt.NDArray, np.ndarray, np.ndarray], None, None]:
    """Gets a generator that yields a tuple with a block, block position and block padding."""

    kernel_radius = _calculate_kernel_radius(sigma, truncate)

    for x0 in range(0, data.shape[0], block_size):
        x1 = x0 + block_size
        for y0 in range(0, data.shape[1], block_size):
            y1 = y0 + block_size
            for z0 in range(0, data.shape[2], block_size):
                z1 = z0 + block_size

                block = data[
                    max(0, x0 - kernel_radius) : x1 + kernel_radius,
                    max(0, y0 - kernel_radius) : y1 + kernel_radius,
                    max(0, z0 - kernel_radius) : z1 + kernel_radius,
                ]

                cx0 = kernel_radius + min(0, x0 - kernel_radius)
                cy0 = kernel_radius + min(0, y0 - kernel_radius)
                cz0 = kernel_radius + min(0, z0 - kernel_radius)
                cx1 = max(0, min(kernel_radius, data.shape[0] - x1))
                cy1 = max(0, min(kernel_radius, data.shape[1] - y1))
                cz1 = max(0, min(kernel_radius, data.shape[2] - z1))

                if copy:
                    block = np.array(block)

                yield block, np.array(((x0, x1), (y0, y1), (z0, z1))), np.array(((cx0, cx1), (cy0, cy1), (cz0, cz1)))


def get_blocks(
    data: npt.NDArray,
    sigma: float,
    block_size: int = 512,
    truncate: float = 4.0,
    copy: bool = False,
) -> tuple[list[npt.NDArray], np.ndarray, np.ndarray]:
    """Gets a tuple of blocks, block positions and block paddings."""

    blocks = []
    block_positions = []
    block_paddings = []

    for block, pos, pad in get_block_generator(data, sigma, block_size=block_size, truncate=truncate, copy=copy):
        blocks.append(block)
        block_positions.append(pos)
        block_paddings.append(pad)

    return blocks, np.array(block_positions), np.array(block_paddings)


def remove_padding(block: npt.NDArray, pad: npt.NDArray[np.integer]) -> npt.NDArray:
    """Slices away the block padding."""

    block = block[
        ...,
        pad[0, 0] : block.shape[-3] - pad[0, 1],
        pad[1, 0] : block.shape[-2] - pad[1, 1],
        pad[2, 0] : block.shape[-1] - pad[2, 1],
    ]
    return block


def insert_block(
    volume: npt.NDArray,
    block: npt.NDArray,
    pos: npt.NDArray[np.integer],
    pad: npt.NDArray[np.integer] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
):
    """Inserts a block into a volume at a specific position."""

    if pad is not None:
        block = remove_padding(block, pad)

    view = volume[..., pos[0, 0] : pos[0, 1], pos[1, 0] : pos[1, 1], pos[2, 0] : pos[2, 1]]

    if cp is not None and isinstance(view, np.ndarray) and isinstance(block, cp.ndarray):
        if volume.dtype != block.dtype:
            block = block.astype(volume.dtype)

        # Move block from GPU to CPU.
        block = cp.asnumpy(block.astype(view.dtype))

    if mask is None:
        view[:] = block
    else:
        view[..., mask] = block
