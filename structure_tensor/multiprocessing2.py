import logging
from dataclasses import dataclass
from multiprocessing import Pool, RawArray, SimpleQueue, cpu_count
from typing import Any, Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt

from . import st3d, util

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    from .cp import st3dcp
except Exception as ex:
    cp = None
    logger.warning("Could not load CuPy: %s", str(ex))


logger.warning(
    "The multiprocessing module is deprecated and will likely be removed in a future version. Please use the multithreading module instead."
)


@dataclass(frozen=True)
class RawArrayArgs:
    array: Any
    shape: tuple[int, ...]
    dtype: npt.DTypeLike

    def get_array(self) -> np.ndarray:
        return np.frombuffer(self.array, dtype=self.dtype).reshape(self.shape)


@dataclass(frozen=True)
class MemoryMapArgs:
    path: str
    shape: tuple[int, ...]
    dtype: npt.DTypeLike
    offset: int
    mode: Literal["r", "r+"]

    def get_array(self) -> np.memmap:
        return np.memmap(self.path, dtype=self.dtype, shape=self.shape, mode=self.mode, offset=self.offset)


@dataclass
class DataSources:
    data: np.ndarray | np.memmap
    structure_tensor: np.ndarray | np.memmap | None
    eigenvectors: np.ndarray | np.memmap | None
    eigenvalues: np.ndarray | np.memmap | None
    device: str


@dataclass(frozen=True)
class InitArgs:
    data_args: RawArrayArgs | MemoryMapArgs
    structure_tensor_args: RawArrayArgs | MemoryMapArgs | None
    eigenvectors_args: RawArrayArgs | MemoryMapArgs | None
    eigenvalues_args: RawArrayArgs | MemoryMapArgs | None
    rho: float
    sigma: float
    block_size: int
    truncate: float
    include_all_eigenvalues: bool
    devices: SimpleQueue[str]

    def get_data_sources(self) -> DataSources:
        return DataSources(
            data=self.data_args.get_array(),
            structure_tensor=self.structure_tensor_args.get_array() if self.structure_tensor_args is not None else None,
            eigenvectors=self.eigenvectors_args.get_array() if self.eigenvectors_args is not None else None,
            eigenvalues=self.eigenvalues_args.get_array() if self.eigenvalues_args is not None else None,
            device=self.devices.get(),
        )


def _create_raw_array(shape: tuple[int, ...], dtype: npt.DTypeLike) -> tuple[Any, np.ndarray]:
    raw = RawArray("b", np.prod(shape, dtype=np.int64).item() * np.dtype(dtype).itemsize)
    a = np.frombuffer(raw, dtype=dtype).reshape(shape)

    return raw, a


def parallel_structure_tensor_analysis(
    volume: np.ndarray | np.memmap,
    sigma: float,
    rho: float,
    eigenvectors: np.memmap | bool = True,
    eigenvalues: np.memmap | bool = True,
    structure_tensor: np.memmap | bool = False,
    truncate: float = 4.0,
    block_size: int = 128,
    include_all_eigenvalues: bool = False,
    devices: Sequence[str] | None = None,
    progress_callback_fn: Callable | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Check that at least one output is specified.
    if all(isinstance(output, bool) and not output for output in [eigenvectors, eigenvalues, structure_tensor]):
        raise ValueError("At least one output must be specified.")

    # Handle input data.
    if isinstance(volume, np.memmap):
        # If memory map, get file path.
        assert volume.filename is not None
        logger.info(
            f"Volume data provided as {str(volume.dtype)} numpy.memmap with shape {volume.shape} occupying {volume.nbytes:,} bytes."
        )
        data_args = MemoryMapArgs(
            path=volume.filename,
            shape=volume.shape,
            dtype=volume.dtype,
            offset=volume.offset,
            mode="r",
        )
    elif isinstance(volume, np.ndarray):
        # If ndarray, copy data to shared memory array. This will double the memory usage.
        # Shared memory can be access by all processes without having to be copied.
        logger.info(
            f"Volume data provided as {str(volume.dtype)} numpy.ndarray with shape {volume.shape} occupying {volume.nbytes:,} bytes."
        )
        volume_raw_array, volume_array = _create_raw_array(volume.shape, volume.dtype)
        volume_array[:] = volume
        data_args = RawArrayArgs(
            array=volume_raw_array,
            shape=volume.shape,
            dtype=volume.dtype,
        )
    else:
        raise ValueError(
            f"Invalid type '{type(volume)}' for volume. Volume must be 'numpy.memmap' and 'numpy.ndarray'."
        )

    # Eigenvector output.
    eigenvectors_shape = (3,) + volume.shape
    eigenvectors_array = None
    eigenvectors_args = None
    if isinstance(eigenvectors, bool):
        if eigenvectors:
            eigenvectors_dtype = np.float32
            eigenvectors_raw_array, eigenvectors_array = _create_raw_array(
                eigenvectors_shape,
                eigenvectors_dtype,
            )
            eigenvectors_args = RawArrayArgs(
                array=eigenvectors_raw_array,
                shape=eigenvectors_shape,
                dtype=eigenvectors_dtype,
            )
    elif isinstance(eigenvectors, np.memmap):
        assert eigenvectors.filename is not None
        eigenvectors_args = MemoryMapArgs(
            path=eigenvectors.filename,
            shape=eigenvectors.shape,
            dtype=eigenvectors.dtype,
            offset=eigenvectors.offset,
            mode="r+",
        )

    assert eigenvectors_array is None or eigenvectors_shape == eigenvectors_array.shape

    # Eigenvalue output.
    if include_all_eigenvalues:
        eigenvalues_shape = (3, 3) + volume.shape
    else:
        eigenvalues_shape = (3,) + volume.shape

    eigenvalues_array = None
    eigenvalues_args = None
    if isinstance(eigenvalues, bool):
        if eigenvalues:
            eigenvalues_dtype = np.float32
            eigenvalues_raw_array, eigenvalues_array = _create_raw_array(
                eigenvalues_shape,
                eigenvalues_dtype,
            )
            eigenvalues_args = RawArrayArgs(
                array=eigenvalues_raw_array,
                shape=eigenvalues_shape,
                dtype=eigenvalues_dtype,
            )
    elif isinstance(eigenvalues, np.memmap):
        assert eigenvalues.filename is not None
        eigenvalues_args = MemoryMapArgs(
            path=eigenvalues.filename,
            shape=eigenvalues.shape,
            dtype=eigenvalues.dtype,
            offset=eigenvalues.offset,
            mode="r+",
        )

    assert eigenvalues_array is None or eigenvalues_shape == eigenvalues_array.shape

    # Structure tensor output.
    structure_tensor_shape = (6,) + volume.shape
    structure_tensor_array = None
    structure_tensor_args = None
    if isinstance(structure_tensor, bool):
        if structure_tensor:
            structure_tensor_dtype = np.float32
            structure_tensor_raw_array, structure_tensor_array = _create_raw_array(
                structure_tensor_shape,
                structure_tensor_dtype,
            )
            structure_tensor_args = RawArrayArgs(
                array=structure_tensor_raw_array,
                shape=structure_tensor_shape,
                dtype=structure_tensor_dtype,
            )
    elif isinstance(structure_tensor, np.memmap):
        assert structure_tensor.filename is not None
        structure_tensor_args = MemoryMapArgs(
            path=structure_tensor.filename,
            shape=structure_tensor.shape,
            dtype=structure_tensor.dtype,
            offset=structure_tensor.offset,
            mode="r+",
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

    init_args = InitArgs(
        data_args=data_args,
        structure_tensor_args=structure_tensor_args,
        eigenvectors_args=eigenvectors_args,
        eigenvalues_args=eigenvalues_args,
        rho=rho,
        sigma=sigma,
        block_size=block_size,
        truncate=truncate,
        include_all_eigenvalues=include_all_eigenvalues,
        devices=queue,
    )

    block_count = util.get_block_count(volume, block_size)
    count = 0
    results = []
    logger.info(f"Volume partitioned into {block_count} blocks.")
    with Pool(processes=len(devices), initializer=init_worker, initargs=(init_args,)) as pool:
        for res in pool.imap_unordered(
            do_work,
            range(block_count),
            chunksize=1,
        ):
            count += 1
            logger.info(f"Block {res} complete ({count}/{block_count}).")
            results.append(res)
            if isinstance(progress_callback_fn, Callable):
                progress_callback_fn(count, block_count)

    output = []

    if structure_tensor_array is not None:
        output.append(structure_tensor_array)

    if eigenvectors_array is not None:
        output.append(eigenvectors_array)

    if eigenvalues_array is not None:
        output.append(eigenvalues_array)

    if len(output) == 1:
        return output[0]
    elif len(output) == 2:
        return output[0], output[1]
    else:
        return output[0], output[1], output[2]


worker_args: InitArgs | None = None
data_sources: DataSources | None = None


def init_worker(init_args: InitArgs):
    """Initialization function for worker."""

    global worker_args
    global data_sources

    worker_args = init_args
    data_sources = init_args.get_data_sources()


def do_work(block_id: int):
    """Worker function."""

    if worker_args is None:
        raise ValueError("Worker not initialized.")

    if data_sources is None:
        raise ValueError("Data sources not initialized.")

    if cp is not None and data_sources.device.startswith("cuda"):
        split = data_sources.device.split(":")
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
        data_sources.data,
        sigma=max(worker_args.sigma, worker_args.rho),
        block_size=worker_args.block_size,
        truncate=worker_args.truncate,
        copy=False,
    )

    # Copy, cast and possibly move data to GPU.
    block = lib.array(block, dtype=np.float64)

    # Calculate structure tensor.
    S = st.structure_tensor_3d(
        block,
        sigma=worker_args.sigma,
        rho=worker_args.rho,
        truncate=worker_args.truncate,
    )

    if data_sources.structure_tensor is not None:
        # Insert S if relevant.
        util.insert_block(data_sources.structure_tensor, S, pos, pad)

    # Calculate eigenvectors and values.
    val, vec = st.eig_special_3d(S, full=worker_args.include_all_eigenvalues)

    if data_sources.eigenvectors is not None:
        # Insert vectors if relevant.
        util.insert_block(data_sources.eigenvectors, vec, pos, pad)

    if data_sources.eigenvalues is not None:
        # Flip so largest value is first.
        val = lib.flip(val, axis=0)

        # Insert values if relevant.
        util.insert_block(data_sources.eigenvalues, val, pos, pad)

    return block_id
