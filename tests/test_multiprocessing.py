import os
import uuid
import pytest
import numpy as np

from structure_tensor import multiprocessing

TEST_FILE_DIR = f".test_multiprocessing"


def get_test_devices():
    devices = [None, ["cpu"] * 4]
    try:
        import cupy

        devices.append(["cuda:0"] * 4)
    except:
        pass
    return devices


@pytest.fixture(scope="session", autouse=True)
def setup():
    """Setup test directory."""
    os.makedirs(TEST_FILE_DIR, exist_ok=True)


@pytest.mark.parametrize("volume_shape", [(50, 59, 51)])
@pytest.mark.parametrize(
    "slices",
    [
        None,
        (slice(0, 11, 2), slice(47, 4, -1), slice(5, -5)),
    ],
)
@pytest.mark.parametrize("transpose", [(-3, -2, -1), (-1, -3, -2), (-2, -1, -3)])
@pytest.mark.parametrize("sigma", [2.5, 10])
@pytest.mark.parametrize("rho", [1.5, 5])
@pytest.mark.parametrize("block_size", [20, 100])
@pytest.mark.parametrize("truncate", [2.1, 4])
@pytest.mark.parametrize("devices", get_test_devices())
@pytest.mark.parametrize("pool_type", ["process", "thread"])
def test_parallel_structure_tensor_analysis(
    volume_shape,
    slices,
    transpose,
    sigma,
    rho,
    block_size,
    truncate,
    devices,
    pool_type,
):
    """Test parallel structure tensor analysis."""

    if slices is None:
        slices = tuple(slice(None) for _ in volume_shape)

    volume = np.random.random(volume_shape).astype(np.float32)[slices]

    try:
        S_0, val_0, vec_0 = multiprocessing.parallel_structure_tensor_analysis(
            volume=volume.transpose(transpose),
            sigma=sigma,
            rho=rho,
            block_size=block_size,
            truncate=truncate,
            devices=devices,
            structure_tensor=np.float32,
            pool_type=pool_type,
        )
    except Exception as ex:
        if pool_type == "thread" and devices and any("cuda:" in d.lower() for d in devices):
            with pytest.raises(ValueError, match="CuPy is not available in thread pools. Use 'process' pool instead."):
                raise ex
            # Skip test the ramaining tests
            return
        else:
            raise ex

    test_volume_name = os.path.join(TEST_FILE_DIR, f"test_volume_{str(uuid.uuid4())}.npy")
    memmap_volume = np.lib.format.open_memmap(
        test_volume_name,
        mode="w+",
        dtype=np.float32,
        shape=volume_shape,
    )[slices]
    memmap_volume[:] = volume

    S_1, val_1, vec_1 = multiprocessing.parallel_structure_tensor_analysis(
        volume=volume.transpose(transpose),
        sigma=sigma,
        rho=rho,
        block_size=block_size,
        truncate=truncate,
        devices=devices,
        structure_tensor=np.float32,
        pool_type=pool_type,
    )

    assert S_0 is not None
    assert val_0 is not None
    assert vec_0 is not None
    assert S_1 is not None
    assert val_1 is not None
    assert vec_1 is not None

    np.testing.assert_array_equal(S_0, S_1)
    np.testing.assert_array_equal(val_0, val_1)
    np.testing.assert_array_equal(vec_0, vec_1)

    test_val_name = os.path.join(TEST_FILE_DIR, f"test_val_{str(uuid.uuid4())}.npy")
    val_1 = np.lib.format.open_memmap(
        test_val_name,
        mode="w+",
        dtype=np.float32,
        shape=(3,) + volume_shape,
    )[(slice(None),) + slices]

    test_vec_name = os.path.join(TEST_FILE_DIR, f"test_vec_{str(uuid.uuid4())}.npy")
    vec_1 = np.lib.format.open_memmap(
        test_vec_name,
        mode="w+",
        dtype=np.float32,
        shape=(3,) + volume_shape,
    )[(slice(None),) + slices]

    test_S1_name = os.path.join(TEST_FILE_DIR, f"test_S1_{str(uuid.uuid4())}.npy")
    S_1 = np.lib.format.open_memmap(
        test_S1_name,
        mode="w+",
        dtype=np.float32,
        shape=(6,) + volume_shape,
    )[(slice(None),) + slices]

    S_1, val_1, vec_1 = multiprocessing.parallel_structure_tensor_analysis(
        volume=volume.transpose(transpose),
        sigma=sigma,
        rho=rho,
        block_size=block_size,
        truncate=truncate,
        devices=devices,
        eigenvalues=val_1.transpose((0,) + transpose),
        eigenvectors=vec_1.transpose((0,) + transpose),
        structure_tensor=S_1.transpose((0,) + transpose),
        pool_type=pool_type,
    )

    assert S_1 is not None
    assert val_1 is not None
    assert vec_1 is not None

    np.testing.assert_array_equal(S_0, S_1)
    np.testing.assert_array_equal(val_0, val_1)
    np.testing.assert_array_equal(vec_0, vec_1)

    del memmap_volume
    del val_1
    del vec_1
    del S_1

    # TODO: Add more assertions
