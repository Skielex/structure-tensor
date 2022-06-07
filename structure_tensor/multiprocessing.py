import logging
from multiprocessing import Pool, RawArray, cpu_count
from types import SimpleNamespace
from typing import Callable

import numpy as np

from . import st3d, util

try:
    import cupy as cp
    from .cp import st3dcp
except Exception as ex:
    cp = None
    logging.warning("Could not load CuPy: %s", str(ex))


def parallel_structure_tensor_analysis(
    volume,
    sigma,
    rho,
    eigenvectors=True,
    eigenvectors_path=None,
    eigenvectors_dtype=np.float32,
    eigenvalues=True,
    eigenvalues_path=None,
    eigenvalues_dtype=np.float32,
    structure_tensor=False,
    structure_tensor_path=None,
    structure_tensor_dtype=np.float32,
    truncate=4.0,
    block_size=128,
    include_all_eigenvalues=False,
    devices=None,
    progress_callback_fn=None,
):

    # Check that at least one output is specified.
    if not any([eigenvectors, eigenvalues, structure_tensor]):
        raise ValueError('At least one output must be specified.')

    # Handle input data.
    data_path = None
    volume_array = None
    if isinstance(volume, np.memmap):
        # If memory map, get file path.
        logging.info(
            f'Volume data provided as {str(volume.dtype)} numpy.memmap with shape {volume.shape} occupying {volume.nbytes:,} bytes.'
        )
        data_path = volume.filename
    elif isinstance(volume, np.ndarray):
        # If ndarray, copy data to shared memory array. This will double the memory usage.
        # Shared memory can be access by all processes without having to be copied.
        logging.info(
            f'Volume data provided as {str(volume.dtype)} numpy.ndarray with shape {volume.shape} occupying {volume.nbytes:,} bytes.'
        )
        volume_array = RawArray('b', volume.nbytes)
        volume_array_np = np.frombuffer(
            volume_array,
            dtype=volume.dtype,
        ).reshape(volume.shape)
        volume_array_np[:] = volume
    else:
        raise ValueError(
            f"Invalid type '{type(volume)}' for volume. Volume must be 'numpy.memmap' and 'numpy.ndarray'."
        )

    # Create list for output.
    output = []

    structure_tensor_shape = None
    structure_tensor_array = None
    # Structure tensor output.
    if structure_tensor:
        structure_tensor_shape = (6, ) + volume.shape

        if structure_tensor_path is None:
            # If no path is set, create shared memory array.
            structure_tensor_array = RawArray(
                'b',
                np.prod(structure_tensor_shape).item() *
                np.dtype(structure_tensor_dtype).itemsize)
            a = np.frombuffer(
                structure_tensor_array,
                dtype=structure_tensor_dtype,
            ).reshape(structure_tensor_shape)
            logging.info(
                f'Created shared memory array for {str(a.dtype)} structure tensor data with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        elif isinstance(structure_tensor_path, str):
            # If path is set, create memory map.
            a = np.memmap(structure_tensor_path,
                          dtype=structure_tensor_dtype,
                          shape=structure_tensor_shape,
                          mode='w+')
            logging.info(
                f'Created memory map at "{eigenvalues_path}" for {str(a.dtype)} structure tensor data with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        else:
            raise ValueError(
                f"Invalid type '{type(structure_tensor_path)}' for structure_tensor_path. Volume must be 'str' or None."
            )

    # Eigenvector output.
    eigenvectors_shape = None
    eigenvectors_array = None
    if eigenvectors:
        eigenvectors_shape = (3, ) + volume.shape

        if eigenvectors_path is None:
            # If no path is set, create shared memory array.
            eigenvectors_array = RawArray(
                'b',
                np.prod(eigenvectors_shape).item() *
                np.dtype(eigenvectors_dtype).itemsize)
            a = np.frombuffer(
                eigenvectors_array,
                dtype=eigenvectors_dtype).reshape(eigenvectors_shape)
            logging.info(
                f'Created shared memory array for {str(a.dtype)} eigenvectors with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        elif isinstance(eigenvectors_path, str):
            # If path is set, create memory map.
            a = np.memmap(eigenvectors_path,
                          dtype=eigenvectors_dtype,
                          shape=eigenvectors_shape,
                          mode='w+')
            logging.info(
                f'Created memory map at "{eigenvalues_path}" for {str(a.dtype)} eigenvectors with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        else:
            raise ValueError(
                f"Invalid type '{type(eigenvectors_path)}' for eigenvector_path. Volume must be 'str' or None."
            )

    # Eigenvalue output.
    eigenvalues_shape = None
    eigenvalues_array = None
    if eigenvalues:
        eigenvalues_shape = (
            3, 3) + volume.shape if include_all_eigenvalues else (
                3, ) + volume.shape

        if eigenvalues_path is None:
            eigenvalues_array = RawArray(
                'b',
                np.prod(eigenvalues_shape).item() *
                np.dtype(eigenvalues_dtype).itemsize)
            a = np.frombuffer(
                eigenvalues_array,
                dtype=eigenvalues_dtype).reshape(eigenvalues_shape)
            logging.info(
                f'Created shared memory array for {str(a.dtype)} eigenvalues with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        elif isinstance(eigenvalues_path, str):
            a = np.memmap(eigenvalues_path,
                          dtype=eigenvalues_dtype,
                          shape=eigenvalues_shape,
                          mode='w+')
            logging.info(
                f'Created memory map at "{eigenvalues_path}" for {str(a.dtype)} eigenvalues with shape {a.shape} occupying {a.nbytes:,} bytes.'
            )
            output.append(a)
        else:
            raise ValueError(
                f"Invalid type '{type(eigenvalues_path)}' for eigenvalues_path. Volume must be 'str' or None."
            )

    # Check devices.
    if devices is None:
        # Use all CPUs.
        devices = ['cpu'] * cpu_count()
    elif all(
            isinstance(d, str) and (d.lower() == 'cpu' or 'cuda:' in d.lower())
            for d in devices):
        pass
    else:
        raise ValueError(
            "Invalid devices. Should be a list of 'cpu' or 'cuda:X', where X is the CUDA device number."
        )
    # As list.
    devices = list(devices)

    # Set arguments.
    init_args = {
        'data_array': volume_array,
        'data_path': data_path,
        'data_dtype': volume.dtype,
        'data_shape': volume.shape,
        'structure_tensor': {
            'array': structure_tensor_array,
            'path': structure_tensor_path,
            'dtype': structure_tensor_dtype,
            'shape': structure_tensor_shape,
        } if structure_tensor else None,
        'eigenvectors': {
            'array': eigenvectors_array,
            'path': eigenvectors_path,
            'dtype': eigenvectors_dtype,
            'shape': eigenvectors_shape,
        } if eigenvectors else None,
        'eigenvalues': {
            'array': eigenvalues_array,
            'path': eigenvalues_path,
            'dtype': eigenvalues_dtype,
            'shape': eigenvalues_shape,
        } if eigenvalues else None,
        'rho': rho,
        'sigma': sigma,
        'block_size': block_size,
        'truncate': truncate,
        'include_all_eigenvalues': include_all_eigenvalues,
        'devices': devices,
    }

    block_count = util.get_block_count(volume, block_size)
    count = 0
    results = []
    logging.info(f'Volume partitioned into {block_count} blocks.')
    with Pool(processes=len(devices),
              initializer=init_worker,
              initargs=(init_args, )) as pool:
        for res in pool.imap_unordered(
                do_work,
                range(block_count),
                chunksize=1,
        ):
            count += 1
            logging.info(f'Block {res} complete ({count}/{block_count}).')
            results.append(res)
            if isinstance(progress_callback_fn, Callable):
                progress_callback_fn(count, block_count)

    # Return output as tuple.
    return tuple(output)


param_dict = {}


def init_worker(kwargs):
    """Initialization function for worker."""

    output_names = ['structure_tensor', 'eigenvectors', 'eigenvalues']

    for k in kwargs:
        if k == 'data_array' and kwargs[k] is not None:
            # Create ndarray from shared memory.
            shared_array = kwargs['data_array']
            param_dict['data'] = np.ndarray(
                buffer=shared_array,
                dtype=kwargs['data_dtype'],
                shape=kwargs['data_shape'],
            )
        elif k == 'data_path' and kwargs[k] is not None:
            # Open read-only memmap.
            param_dict['data'] = np.memmap(kwargs['data_path'],
                                           dtype=kwargs['data_dtype'],
                                           shape=kwargs['data_shape'],
                                           mode='r')
        elif k in output_names and kwargs[k] is not None:
            d = kwargs[k]
            if d['array'] is not None:
                # Create ndarray from shared memory.
                shared_array = d['array']
                param_dict[k] = np.ndarray(
                    buffer=shared_array,
                    dtype=d['dtype'],
                    shape=d['shape'],
                )
            elif d['path'] is not None:
                # Open read/write memmap.
                param_dict[k] = np.memmap(d['path'],
                                          dtype=d['dtype'],
                                          shape=d['shape'],
                                          mode='r+')
        else:
            param_dict[k] = kwargs[k]


def do_work(block_id):
    """Worker function."""

    params = SimpleNamespace(**param_dict)

    if isinstance(params.devices, list):
        # If more devices are provided select one.
        params.devices = params.devices[block_id % len(params.devices)]
        # Overwrite initial device value to prevent process changing device on next iteration.
        param_dict['devices'] = params.devices

    if cp is not None and isinstance(
            params.devices, str) and params.devices.startswith('cuda'):

        split = params.devices.split(':')
        if len(split) > 1:
            # Overwrite initial device value to prevent process changing device on next iteration.
            param_dict['devices'] = split[0]

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
        params.data,
        sigma=max(params.sigma, params.rho),
        block_size=params.block_size,
        truncate=params.truncate,
        copy=False,
    )

    # Copy, cast and possibly move data to GPU.
    block = lib.array(block, dtype=np.float64)

    # Calculate structure tensor.
    S = st.structure_tensor_3d(
        block,
        sigma=params.sigma,
        rho=params.rho,
        truncate=params.truncate,
    )

    if params.structure_tensor is not None:
        # Insert S if relevant.
        util.insert_block(params.structure_tensor, S, pos, pad)

    # Calculate eigenvectors and values.
    val, vec = st.eig_special_3d(S, full=params.include_all_eigenvalues)

    if params.eigenvectors is not None:
        # Insert vectors if relevant.
        util.insert_block(params.eigenvectors, vec, pos, pad)

    if params.eigenvalues is not None:
        # Flip so largest value is first.
        val = lib.flip(val, axis=0)

        # Insert values if relevant.
        util.insert_block(params.eigenvalues, val, pos, pad)

    return block_id
