"""Module for parallel structure tensor analysis using multi-processing."""

import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool, RawArray, SimpleQueue, cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Literal, Sequence, Union

import numpy as np
import numpy.typing as npt

from . import st3d, util

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    from .cp import st3dcp

    _cupy_import_error = None
except Exception as ex:
    cp = None
    st3dcp = None
    _cupy_import_error = ex


DEFAULT_POOL_TYPE = "thread" if os.name == "nt" else "process"


@dataclass(frozen=True)
class _ArrayArgs:
    array: np.ndarray

    def get_array(self) -> np.ndarray:
        return self.array


@dataclass(frozen=True)
class _RawArrayArgs:
    array: Any
    shape: tuple[int, ...]
    dtype: npt.DTypeLike

    def get_array(self) -> np.ndarray:
        return np.frombuffer(self.array, dtype=self.dtype).reshape(self.shape)


@dataclass(frozen=True)
class _StrideInfo:
    strides: tuple[int, ...]
    offset: int

    @staticmethod
    def from_memmap(m: np.memmap) -> Union["_StrideInfo", None]:
        if not isinstance(m.base, np.memmap):
            return None

        return _StrideInfo(
            strides=m.strides,
            offset=m.ctypes.data - m.base.ctypes.data,
        )


@dataclass(frozen=True)
class _MemoryMapArgs:
    path: str
    shape: tuple[int, ...]
    dtype: npt.DTypeLike
    offset: int
    mode: Literal["r", "r+"]
    stride_info: _StrideInfo | None = None

    @staticmethod
    def from_memmap(m: np.memmap, mode: Literal["r", "r+"]) -> "_MemoryMapArgs":
        assert m.filename is not None
        return _MemoryMapArgs(
            path=m.filename,
            shape=m.shape,
            dtype=m.dtype,
            offset=m.offset,
            mode=mode,
            stride_info=_StrideInfo.from_memmap(m),
        )

    def get_array(self) -> np.ndarray:
        if self.stride_info is None:
            return np.memmap(self.path, dtype=self.dtype, shape=self.shape, mode=self.mode, offset=self.offset)
        else:
            map = np.memmap(
                self.path,
                mode=self.mode,
                offset=self.offset,
            )
            map = map[self.stride_info.offset :].view(self.dtype)
            map = np.lib.stride_tricks.as_strided(map, shape=self.shape, strides=self.stride_info.strides)
            return map


@dataclass
class _DataSources:
    data: np.ndarray | np.memmap
    structure_tensor: np.ndarray | np.memmap | None
    eigenvectors: np.ndarray | np.memmap | None
    eigenvalues: np.ndarray | np.memmap | None
    device: str


@dataclass(frozen=True)
class _InitArgs:
    data_args: _RawArrayArgs | _MemoryMapArgs | _ArrayArgs
    structure_tensor_args: _RawArrayArgs | _MemoryMapArgs | _ArrayArgs | None
    eigenvectors_args: _RawArrayArgs | _MemoryMapArgs | _ArrayArgs | None
    eigenvalues_args: _RawArrayArgs | _MemoryMapArgs | _ArrayArgs | None
    rho: float
    sigma: float
    block_size: int
    truncate: float
    include_all_eigenvalues: bool
    eigenvalue_order: Literal["desc", "asc"]
    devices: SimpleQueue

    def get_data_sources(self) -> _DataSources:
        return _DataSources(
            data=self.data_args.get_array(),
            structure_tensor=self.structure_tensor_args.get_array() if self.structure_tensor_args is not None else None,
            eigenvectors=self.eigenvectors_args.get_array() if self.eigenvectors_args is not None else None,
            eigenvalues=self.eigenvalues_args.get_array() if self.eigenvalues_args is not None else None,
            device=self.devices.get(),
        )


def _create_raw_array(shape: tuple[int, ...], dtype: npt.DTypeLike) -> tuple[Any, np.ndarray]:
    raw = RawArray("b", np.prod(np.asarray(shape), dtype=np.int64).item() * np.dtype(dtype).itemsize)
    a = np.frombuffer(raw, dtype=dtype).reshape(shape)

    return raw, a


def parallel_structure_tensor_analysis(
    volume: np.ndarray,
    sigma: float,
    rho: float,
    eigenvectors: np.memmap | npt.DTypeLike | None = np.float32,
    eigenvalues: np.memmap | npt.DTypeLike | None = np.float32,
    structure_tensor: np.memmap | npt.DTypeLike | None = None,
    truncate: float = 4.0,
    block_size: int = 128,
    include_all_eigenvectors: bool = False,
    eigenvalue_order: Literal["desc", "asc"] = "desc",
    devices: Sequence[str] | None = None,
    progress_callback_fn: Callable[[int, int], None] | None = None,
    fallback_to_cpu: bool = True,
    pool_type: Literal["process", "thread"] = DEFAULT_POOL_TYPE,
):
    """Perform parallel structure tensor analysis on a 3D volume. Returns the structure tensor, eigenvalues, and eigenvectors.

    Args:
        volume: The 3D volume to analyze.
        sigma: The standard deviation of the Gaussian kernel used for smoothing.
        rho: The standard deviation of the Gaussian kernel used for integration.
        eigenvectors: The output array for the eigenvectors. If a dtype is provided, a new array is created. If None, eigenvectors are not returned.
        eigenvalues: The output array for the eigenvalues. If a dtype is provided, a new array is created. If None, eigenvalues are not returned.
        structure_tensor: The output array for the structure tensor. If a dtype is provided, a new array is created. If None, the structure tensor is not returned.
        truncate: The number of standard deviations to truncate the Gaussian kernel.
        block_size: The size of the blocks to process.
        include_all_eigenvectors: Whether to include all eigenvectors or just the vector corresponding to the smallet eigenvalue.
        eigenvalue_order: The order of eigenvalues. Either "desc" for descending or "asc" for ascending. If all three eigenvectors are returned, they will be ordered according to the eigenvalues.
        devices: The devices to use for processing. May one or more instances of "cpu" or "cuda:X", where X is the CUDA device number. For example `["cpu", "cpu", "cuda:0"]` will use use two CPU-based processes and one GPU-based process running on CUDA device 1. If None, all one CPU-based process per CPU core is used.
        progress_callback_fn: A callback function that is called with the current block count and the total block count.
        fallback_to_cpu: Whether to fall back to CPU if CuPy is specified but is not available.
        pool_type: The type of pool to use. Either "process" or "thread". If "process", a process pool is used. If "thread", a thread pool is used. The default is "thread" on Windows and "process" on other platforms.

    Returns:
        A tuple containing the structure tensor, eigenvalues, and eigenvectors. Each of the value may be None if not requested.
    """

    if pool_type not in ["process", "thread"]:
        raise ValueError("Invalid pool type. Should be 'process' or 'thread'.")

    if pool_type == "thread" and devices and any("cuda:" in d.lower() for d in devices):
        raise ValueError("CuPy is not available in thread pools. Use 'process' pool instead.")

    use_process_pool = pool_type == "process"
    copy_to_raw_array = use_process_pool and os.name == "nt"

    # Check that at least one output is specified.
    if all(output is None for output in [eigenvectors, eigenvalues, structure_tensor]):
        raise ValueError("At least one output must be specified.")

    if devices and _cupy_import_error is not None and any("cuda:" in d.lower() for d in devices):
        if fallback_to_cpu:
            logger.warning("CuPy not available. Falling back to NumPy.")
        else:
            raise _cupy_import_error

    logger.info(
        f"Volume data provided as {str(volume.dtype)} {type(volume)} with shape {volume.shape} occupying {volume.nbytes:,} bytes."
    )
    # Handle input data.

    if copy_to_raw_array and isinstance(volume, np.memmap):
        # If memory map, get file path.
        assert volume.filename is not None
        data_args = _MemoryMapArgs.from_memmap(volume, mode="r")
    elif copy_to_raw_array and isinstance(volume, np.ndarray):
        # If ndarray, copy data to shared memory array. This will double the memory usage.
        # Shared memory can be access by all processes without having to be copied.
        volume_raw_array, volume_array = _create_raw_array(volume.shape, volume.dtype)
        volume_array[:] = volume
        data_args = _RawArrayArgs(
            array=volume_raw_array,
            shape=volume.shape,
            dtype=volume.dtype,
        )
    elif not copy_to_raw_array and isinstance(volume, np.ndarray):
        # If ndarray or memmap and using thread pool, use as is.
        data_args = _ArrayArgs(array=volume)
    else:
        raise ValueError(f"Invalid type '{type(volume)}' for volume. Volume must be an 'numpy.ndarray'.")

    # Eigenvector output.
    if include_all_eigenvectors:
        eigenvectors_shape = (3, 3) + volume.shape
    else:
        eigenvectors_shape = (3,) + volume.shape
    eigenvectors_array = None
    eigenvectors_args = None

    if eigenvectors is None:
        pass
    elif copy_to_raw_array and isinstance(eigenvectors, np.memmap):
        assert eigenvectors.filename is not None
        eigenvectors_args = _MemoryMapArgs.from_memmap(eigenvectors, mode="r+")
        eigenvectors_array = eigenvectors
    elif not copy_to_raw_array and isinstance(eigenvectors, np.memmap):
        eigenvectors_args = _ArrayArgs(array=eigenvectors)
        eigenvectors_array = eigenvectors
    else:
        eigenvectors_dtype = eigenvectors
        eigenvectors_raw_array, eigenvectors_array = _create_raw_array(
            eigenvectors_shape,
            eigenvectors_dtype,
        )
        eigenvectors_args = _RawArrayArgs(
            array=eigenvectors_raw_array,
            shape=eigenvectors_shape,
            dtype=eigenvectors_dtype,
        )

    assert eigenvectors_array is None or eigenvectors_shape == eigenvectors_array.shape

    # Eigenvalue output.
    eigenvalues_shape = (3,) + volume.shape
    eigenvalues_array = None
    eigenvalues_args = None

    if eigenvalues is None:
        pass
    elif copy_to_raw_array and isinstance(eigenvalues, np.memmap):
        assert eigenvalues.filename is not None
        eigenvalues_args = _MemoryMapArgs.from_memmap(eigenvalues, mode="r+")
        eigenvalues_array = eigenvalues
    elif not copy_to_raw_array and isinstance(eigenvalues, np.memmap):
        eigenvalues_args = _ArrayArgs(array=eigenvalues)
        eigenvalues_array = eigenvalues
    else:
        eigenvalues_dtype = eigenvalues
        eigenvalues_raw_array, eigenvalues_array = _create_raw_array(
            eigenvalues_shape,
            eigenvalues_dtype,
        )
        eigenvalues_args = _RawArrayArgs(
            array=eigenvalues_raw_array,
            shape=eigenvalues_shape,
            dtype=eigenvalues_dtype,
        )

    assert eigenvalues_array is None or eigenvalues_shape == eigenvalues_array.shape

    # Structure tensor output.
    structure_tensor_shape = (6,) + volume.shape
    structure_tensor_array = None
    structure_tensor_args = None

    if structure_tensor is None:
        pass
    elif copy_to_raw_array and isinstance(structure_tensor, np.memmap):
        assert structure_tensor.filename is not None
        structure_tensor_args = _MemoryMapArgs.from_memmap(structure_tensor, mode="r+")
        structure_tensor_array = structure_tensor
    elif not copy_to_raw_array and isinstance(structure_tensor, np.memmap):
        structure_tensor_args = _ArrayArgs(array=structure_tensor)
        structure_tensor_array = structure_tensor
    else:
        structure_tensor_dtype = structure_tensor
        structure_tensor_raw_array, structure_tensor_array = _create_raw_array(
            structure_tensor_shape,
            structure_tensor_dtype,
        )
        structure_tensor_args = _RawArrayArgs(
            array=structure_tensor_raw_array,
            shape=structure_tensor_shape,
            dtype=structure_tensor_dtype,
        )

    # Check devices.
    if devices is None:
        # Use all CPUs.
        devices = ["cpu"] * cpu_count()
    elif all(isinstance(d, str) and (d.lower() == "cpu" or "cuda:" in d.lower()) for d in devices):
        pass
    else:
        raise ValueError("Invalid devices. Should be a list of 'cpu' or 'cuda:X', where X is the CUDA device number.")

    queue = SimpleQueue()
    for device in devices:
        queue.put(device)

    init_args = _InitArgs(
        data_args=data_args,
        structure_tensor_args=structure_tensor_args,
        eigenvectors_args=eigenvectors_args,
        eigenvalues_args=eigenvalues_args,
        rho=rho,
        sigma=sigma,
        block_size=block_size,
        truncate=truncate,
        include_all_eigenvalues=include_all_eigenvectors,
        eigenvalue_order=eigenvalue_order,
        devices=queue,
    )

    block_count = util.get_block_count(volume, block_size)
    count = 0
    results = []
    logger.info(f"Volume partitioned into {block_count} blocks.")
    pool_ctor = Pool if use_process_pool else ThreadPool
    with pool_ctor(processes=len(devices), initializer=_init_worker, initargs=(init_args,)) as pool:
        for res in pool.imap_unordered(
            _do_work,
            range(block_count),
            chunksize=1,
        ):
            count += 1
            logger.info(f"Block {res} complete ({count}/{block_count}).")
            results.append(res)
            if isinstance(progress_callback_fn, Callable):
                progress_callback_fn(count, block_count)

    return structure_tensor_array, eigenvalues_array, eigenvectors_array


_worker_args: _InitArgs | None = None
_data_sources: _DataSources | None = None


def _init_worker(init_args: _InitArgs):
    """Initialization function for worker."""

    global _worker_args
    global _data_sources

    _worker_args = init_args
    _data_sources = init_args.get_data_sources()


def _do_work(block_id: int):
    """Worker function."""

    if _worker_args is None:
        raise ValueError("Worker not initialized.")

    if _data_sources is None:
        raise ValueError("Data sources not initialized.")

    if cp is not None and st3dcp is not None and _data_sources.device.startswith("cuda"):
        split = _data_sources.device.split(":")
        if len(split) > 1:
            # CUDA device ID specified. Use that device.
            device_id = int(split[1])
            cp.cuda.Device(device_id).use()

        # Use CuPy.
        st = st3dcp
        lib = cp
    else:
        # Use NumPy.
        st = st3d
        lib = np

    # Get block, positions and padding.
    block, pos, pad = util.get_block(
        block_id,
        _data_sources.data,
        sigma=max(_worker_args.sigma, _worker_args.rho),
        block_size=_worker_args.block_size,
        truncate=_worker_args.truncate,
        copy=False,
    )

    # Copy, cast and possibly move data to GPU.
    block = lib.array(block, dtype=np.float64)

    # Calculate structure tensor.
    S = st.structure_tensor_3d(
        block,
        sigma=_worker_args.sigma,
        rho=_worker_args.rho,
        truncate=_worker_args.truncate,
    )

    if _data_sources.structure_tensor is not None:
        # Insert S if relevant.
        util.insert_block(_data_sources.structure_tensor, S, pos, pad)

    # Calculate eigenvectors and values.
    val, vec = st.eig_special_3d(
        S,
        full=_worker_args.include_all_eigenvalues,
        eigenvalue_order=_worker_args.eigenvalue_order,
    )

    if _data_sources.eigenvectors is not None:
        # Insert vectors if relevant.
        util.insert_block(_data_sources.eigenvectors, vec, pos, pad)

    if _data_sources.eigenvalues is not None:
        # Insert values if relevant.
        util.insert_block(_data_sources.eigenvalues, val, pos, pad)

    return block_id
