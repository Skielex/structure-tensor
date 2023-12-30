import logging
import threading
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from . import st3d, util

try:
    import cupy as cp

    from .cp import st3dcp
except Exception as ex:
    cp = None
    logging.warning("Could not load CuPy: %s", str(ex))


def parallel_structure_tensor_analysis(
    volume: npt.NDArray[np.number],
    sigma: float,
    rho: float,
    eigenvectors: Optional[npt.NDArray[np.number] | bool] = True,
    eigenvalues: Optional[npt.NDArray[np.number] | bool] = True,
    structure_tensor: Optional[npt.NDArray[np.number] | bool] = None,
    truncate: float = 4.0,
    block_size: int = 128,
    include_all_eigenvalues: bool = False,
    devices: Optional[Sequence[str]] = None,
    progress_callback_fn: Optional[Callable] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    # Check devices.
    if devices is None:
        # Use all CPUs.
        devices = ["cpu"] * cpu_count()
    elif all(isinstance(d, str) and (d.lower() == "cpu" or "cuda" in d.lower()) for d in devices):
        pass
    else:
        raise ValueError("Invalid devices. Should be a list of 'cpu' or 'cuda:X', where X is the CUDA device number.")

    # Devices as list.
    devices = list(devices)[::-1]

    if cp is None:
        non_cuda_devices = [d for d in devices if not d.lower().startswith("cuda")]
        if len(non_cuda_devices) != len(devices):
            devices = non_cuda_devices
            logging.warning("CuPy could not be loaded. Ignoring specified CUDA devices.")

        # Check if devices is not empty.
        if not devices:
            raise ValueError("No valid devices specified.")

    result = []

    dtype = volume.dtype if np.issubdtype(volume.dtype, np.floating) else np.float32

    if isinstance(eigenvectors, bool):
        eigenvectors = np.empty((3,) + volume.shape, dtype=dtype) if eigenvectors else None

    if isinstance(eigenvalues, bool):
        if eigenvalues:
            if include_all_eigenvalues:
                eigenvalues = np.empty((3, 3) + volume.shape, dtype=dtype)
            else:
                eigenvalues = np.empty((3,) + volume.shape, dtype=dtype)
        else:
            eigenvalues = None
    if isinstance(structure_tensor, bool):
        structure_tensor = np.empty((6,) + volume.shape, dtype=dtype) if structure_tensor else None

    result.append(structure_tensor)
    result.append(eigenvectors)
    result.append(eigenvalues)

    # Create block memory views.
    blocks, positions, paddings = util.get_blocks(
        volume,
        block_size=block_size,
        sigma=max(sigma, rho),
        truncate=truncate,
        copy=False,
    )
    block_count = len(blocks)
    logging.info(f"Volume partitioned into {block_count} blocks.")

    if isinstance(progress_callback_fn, Callable):
        progress_callback_fn(0, block_count)

    count = 0
    thread_devices = {}

    pool_args = [
        (
            (block, pos, pad),
            devices,
            thread_devices,
            sigma,
            rho,
            truncate,
            eigenvectors,
            eigenvalues,
            structure_tensor,
            include_all_eigenvalues,
        )
        for block, pos, pad in zip(blocks, positions, paddings)
    ]

    cuda_devices = set()

    with ThreadPool(processes=len(devices)) as pool:
        for thread_id, device_id in pool.imap_unordered(
            _do_work,
            pool_args,
            chunksize=1,
        ):
            count += 1
            logging.info(f"Thread {thread_id} completed block ({count}/{block_count}).")
            if isinstance(progress_callback_fn, Callable):
                progress_callback_fn(count, block_count)
            cuda_devices.add(device_id)

    if cp is not None:
        # Free GPU memory.
        for d in cuda_devices:
            if d is not None:
                with cp.cuda.Device(d):
                    mempool = cp.get_default_memory_pool()
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()

    return tuple(result)


def _do_work(args):
    """Worker function."""

    (
        (block, pos, pad),
        devices,
        thread_devices,
        sigma,
        rho,
        truncate,
        eigenvectors,
        eigenvalues,
        structure_tensor,
        include_all_eigenvalues,
    ) = args

    thread_id = threading.get_ident()
    if thread_id not in thread_devices:
        # print(devices)
        thread_devices[thread_id] = devices.pop()
    device = thread_devices[thread_id]
    device_id = None

    if cp is not None and device.startswith("cuda"):
        # Use CuPy.
        st = st3dcp
        lib = cp

        split = device.split(":")
        if len(split) > 1:
            # CUDA device ID specified. Use that device.
            device_id = int(split[1])
        else:
            device_id = 0
    else:
        # Use NumPy.
        st = st3d
        lib = np

    if cp is not None and device.startswith("cuda"):
        with cp.cuda.Device(device_id):
            _do_work_innner(lib, st, args)
    else:
        _do_work_innner(lib, st, args)

    return thread_id, device_id


def _do_work_innner(lib, st, args):
    (
        (block, pos, pad),
        devices,
        thread_devices,
        sigma,
        rho,
        truncate,
        eigenvectors,
        eigenvalues,
        structure_tensor,
        include_all_eigenvalues,
    ) = args

    # Copy, cast and possibly move data to GPU.
    block = lib.array(block, dtype=np.float64)

    # Calculate structure tensor.
    S = st.structure_tensor_3d(
        block,
        sigma=sigma,
        rho=rho,
        truncate=truncate,
    )

    if structure_tensor is not None:
        # Insert S if relevant.
        util.insert_block(structure_tensor, S, pos, pad)

    # Calculate eigenvectors and values.
    val, vec = st.eig_special_3d(S, full=include_all_eigenvalues)

    if eigenvectors is not None:
        # Insert vectors if relevant.
        util.insert_block(eigenvectors, vec, pos, pad)

    if eigenvalues is not None:
        # Flip so largest value is first.
        val = lib.flip(val, axis=0)

        # Insert values if relevant.
        util.insert_block(eigenvalues, val, pos, pad)
