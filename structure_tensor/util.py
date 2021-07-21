"""Utilities module."""
import numpy as np

try:
    import cupy as cp
except Exception as ex:
    cp = None


def get_block_count(data, block_size=512):
    """Gets the number of blocks that will be created for the given input."""

    return np.prod(np.ceil(np.array(data.shape) / block_size).astype(int))


def get_block(i, data, sigma, block_size=512, truncate=4.0, copy=False):
    """Gets the ith block."""

    kernel_radius = int(sigma * truncate + 0.5)
    count = 0

    for x0 in range(0, data.shape[0], block_size):
        for y0 in range(0, data.shape[1], block_size):
            for z0 in range(0, data.shape[2], block_size):
                if count == i:
                    x1 = x0 + block_size
                    y1 = y0 + block_size
                    z1 = z0 + block_size

                    block = data[max(0, x0 - kernel_radius):x1 + kernel_radius,
                                 max(0, y0 - kernel_radius):y1 + kernel_radius,
                                 max(0, z0 - kernel_radius):z1 + kernel_radius]

                    cx0 = kernel_radius + min(0, x0 - kernel_radius)
                    cy0 = kernel_radius + min(0, y0 - kernel_radius)
                    cz0 = kernel_radius + min(0, z0 - kernel_radius)
                    cx1 = max(0, min(kernel_radius, data.shape[0] - x1))
                    cy1 = max(0, min(kernel_radius, data.shape[1] - y1))
                    cz1 = max(0, min(kernel_radius, data.shape[2] - z1))

                    if copy:
                        block = np.array(block)

                    return block, np.array(
                        ((x0, x1), (y0, y1), (z0, z1))), np.array(
                            ((cx0, cx1), (cy0, cy1), (cz0, cz1)))

                count += 1

    raise IndexError(f"Index {i} is out of bounds for {count} blocks.")


def get_block_generator(data, sigma, block_size=512, truncate=4.0, copy=False):
    """Gets a generator that yields a tuple with a block, block position and block padding."""

    kernel_radius = int(sigma * truncate + 0.5)

    for x0 in range(0, data.shape[0], block_size):
        x1 = x0 + block_size
        for y0 in range(0, data.shape[1], block_size):
            y1 = y0 + block_size
            for z0 in range(0, data.shape[2], block_size):
                z1 = z0 + block_size

                block = data[max(0, x0 - kernel_radius):x1 + kernel_radius,
                             max(0, y0 - kernel_radius):y1 + kernel_radius,
                             max(0, z0 - kernel_radius):z1 + kernel_radius]

                cx0 = kernel_radius + min(0, x0 - kernel_radius)
                cy0 = kernel_radius + min(0, y0 - kernel_radius)
                cz0 = kernel_radius + min(0, z0 - kernel_radius)
                cx1 = max(0, min(kernel_radius, data.shape[0] - x1))
                cy1 = max(0, min(kernel_radius, data.shape[1] - y1))
                cz1 = max(0, min(kernel_radius, data.shape[2] - z1))

                if copy:
                    block = np.array(block)

                yield block, np.array(
                    ((x0, x1), (y0, y1), (z0, z1))), np.array(
                        ((cx0, cx1), (cy0, cy1), (cz0, cz1)))


def get_blocks(data, sigma, block_size=512, truncate=4.0, copy=False):
    """Gets a tuple of blocks, block positions and block paddings."""

    blocks = []
    block_positions = []
    block_paddings = []

    for block, pos, pad in get_block_generator(data,
                                               sigma,
                                               block_size=block_size,
                                               truncate=truncate,
                                               copy=copy):
        blocks.append(block)
        block_positions.append(pos)
        block_paddings.append(pad)

    return blocks, np.array(block_positions), np.array(block_paddings)


def remove_padding(block, pad):
    """Slices away the block padding."""

    block = block[..., pad[0, 0]:block.shape[-3] - pad[0, 1],
                  pad[1, 0]:block.shape[-2] - pad[1, 1],
                  pad[2, 0]:block.shape[-1] - pad[2, 1]]
    return block


def remove_boundary(block, pad, sigma, truncate=4.0):
    """Slices away the parts of the block affected by the boundary.
    
    The goal here is to remove parts of the block that would be affected by 
    insufficient padding, e.g., near the edge of the original volume.
    If the padding was sufficient to avoid boundary artifacts nothing is removed.
    """

    kernel_radius = int(sigma * truncate + 0.5)
    boundary = np.maximum(0, kernel_radius - pad)
    block = block[boundary[0, 0]:block.shape[0] - boundary[0, 1],
                  boundary[1, 0]:block.shape[1] - boundary[1, 1],
                  boundary[2, 0]:block.shape[2] - boundary[2, 1]]
    return block


def insert_block(volume, block, pos, pad=None, mask=None):
    """Inserts a block into a volume at a specific position."""

    if pad is not None:
        block = remove_padding(block, pad)

    view = volume[..., pos[0, 0]:pos[0, 1], pos[1, 0]:pos[1, 1],
                  pos[2, 0]:pos[2, 1]]

    if cp is not None and isinstance(view, np.ndarray) and isinstance(
            block, cp.ndarray):
        # Move block from GPU to CPU.
        block = block.get()

    if mask is None:
        view[:] = block
    else:
        view[..., mask] = block
