import logging
from multiprocessing import Pool, RawArray, cpu_count
from types import SimpleNamespace

import numpy as np

from . import st3d, util

try:
    import cupy as cp
    from .cp import st3dcp
except Exception as ex:
    cp = None
    logging.warning("Could not load CuPy for structure tensor analysis: %s",
                    str(ex))


def parallel_structure_tensor_analysis(
    volume,
    sigma,
    rho,
    output_path=None,
    truncate=4.0,
    block_size=512,
    devices=None,
):

    # Handle input data.
    data_path = None
    volume_array = None
    if isinstance(volume, np.memmap):
        data_path = volume.filename
    elif isinstance(volume, np.ndarray):
        # Copy data to shared memory array. This will double the memory usage
        volume_array = RawArray(np.ctypeslib.as_ctypes_type(volume.dtype),
                                volume.size)
        volume_array_np = np.frombuffer(volume_array, dtype=volume.dtype).reshape(volume.shape)
        volume_array_np[:] = volume
    else:
        raise ValueError(
            f"Invalid type '{type(volume)}' for volume. Volume must be 'numpy.memmap' and 'numpy.ndarray'."
        )

    # Handle output data.
    output_array = None
    if output_path is None:
        output_array = RawArray(np.ctypeslib.as_ctypes_type(volume.dtype),
                                volume.size)
        output = np.frombuffer(output_array,
                               dtype=volume.dtype).reshape(volume.shape)
    elif isinstance(output_path, str):
        output = np.memmap(output_path,
                           dtype=volume.dtype,
                           shape=volume.shape,
                           mode='w+')
    else:
        raise ValueError(
            f"Invalid type '{type(output_path)}' for output_path. Volume must be 'str' or None."
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
            "Invalid devices. Should be 'cpu' or 'cuda:X', where X is the CUDA device number."
        )

    # Set arguments.
    init_args = {
        'data_array': volume_array,
        'data_path': data_path,
        'data_dtype': volume.dtype,
        'data_shape': volume.shape,
        'output_array': output_array,
        'output_path': output_path,
        'output_dtype': volume.dtype,
        'output_shape': volume.shape,
        'rho': rho,
        'sigma': sigma,
        'block_size': block_size,
        'truncate': truncate,
        'devices': devices,
    }

    results = []
    with Pool(processes=len(devices),
              initializer=init_worker,
              initargs=(init_args, )) as pool:
        for res in pool.imap_unordered(
                do_work,
                range(util.get_block_count(volume, block_size)),
                chunksize=1,
        ):
            results.append(res)

    return output


param_dict = {}


def init_worker(kwargs):
    """Initialization function for worker."""

    for k in kwargs:
        if k == 'data_array' and kwargs[k] is not None:
            # Create ndarray from shared memory.
            param_dict['data'] = np.frombuffer(
                kwargs['data_array'],
                dtype=kwargs['data_dtype']).reshape(kwargs['data_shape'])
        elif k == 'data_path' and kwargs[k] is not None:
            # Open read-only memmap.
            param_dict['data'] = np.memmap(kwargs['data_path'],
                                           dtype=kwargs['data_dtype'],
                                           shape=kwargs['data_shape'],
                                           mode='r')
        elif k == 'output_array' and kwargs[k] is not None:
            # Create ndarray from shared memory.
            param_dict['output'] = np.frombuffer(
                kwargs['output_array'],
                dtype=kwargs['output_dtype']).reshape(kwargs['output_shape'])
        elif k == 'output_path' and kwargs[k] is not None:
            # Open read/write memmap.
            param_dict['output'] = np.memmap(kwargs['output_path'],
                                             dtype=kwargs['output_dtype'],
                                             shape=kwargs['output_shape'],
                                             mode='r+')

        param_dict[k] = kwargs[k]


def do_work(block_id):
    """Worker function."""

    params = SimpleNamespace(**param_dict)

    if isinstance(params.devices, list) or isinstance(params.devices, tuple):
        # If more devices are provided select one.
        devices = params.devices[block_id % len(params.devices)]
        # Overwrite initial device value to prevent process changing device on next iteration.
        param_dict['devices'] = devices

    if cp is not None and isinstance(devices,
                                     str) and devices.startswith('cuda'):

        split = devices.split(':')
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

    block, pos, pad = util.get_block(
        block_id,
        params.data,
        sigma=max(params.sigma, params.rho),
        block_size=params.block_size,
        truncate=params.truncate,
        copy=False,
    )

    S = st.structure_tensor_3d(
        block,
        sigma=params.sigma,
        rho=params.rho,
        truncate=params.truncate,
    )

    util.insert_block(params.output, S, pos, pad)
