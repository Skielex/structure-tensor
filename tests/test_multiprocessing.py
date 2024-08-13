import os
import uuid
import pytest
import numpy as np

from structure_tensor import multiprocessing

TEST_FILE_DIR = f".test_multiprocessing"


@pytest.fixture(scope="session", autouse=True)
def setup():
    """Setup test directory."""
    os.makedirs(TEST_FILE_DIR, exist_ok=True)


@pytest.mark.parametrize("volume_shape", [(50, 50, 50), (101, 100, 51)])
@pytest.mark.parametrize("slices", [None, (slice(0, 25), slice(4, 47), slice(1, 40))])
@pytest.mark.parametrize("sigma", [2.5, 10])
@pytest.mark.parametrize("rho", [1.5, 5])
@pytest.mark.parametrize("block_size", [10, 100])
@pytest.mark.parametrize("truncate", [2.0, 4])
@pytest.mark.parametrize("devices", [["cpu"] * 4, ["cuda:0"]])
def test_parallel_structure_tensor_analysis(
    volume_shape,
    slices,
    sigma,
    rho,
    block_size,
    truncate,
    devices,
):
    """Test parallel structure tensor analysis."""

    if slices is None:
        slices = tuple(slice(None) for _ in volume_shape)

    volume = np.random.random(volume_shape).astype(np.float32)[slices]

    S_0, val_0, vec_0 = multiprocessing.parallel_structure_tensor_analysis(
        volume=volume,
        sigma=sigma,
        rho=rho,
        block_size=block_size,
        truncate=truncate,
        devices=devices,
        structure_tensor=np.float32,
    )

    test_volume_name = os.path.join(TEST_FILE_DIR, f"test_volume_{str(uuid.uuid4())}.npy")
    memmap_volume = np.lib.format.open_memmap(
        test_volume_name,
        mode="w+",
        dtype=np.float32,
        shape=volume_shape,
    )[slices]
    memmap_volume[:] = volume

    S_1, val_1, vec_1 = multiprocessing.parallel_structure_tensor_analysis(
        volume=volume,
        sigma=sigma,
        rho=rho,
        block_size=block_size,
        truncate=truncate,
        devices=devices,
        structure_tensor=np.float32,
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
        volume=volume,
        sigma=sigma,
        rho=rho,
        block_size=block_size,
        truncate=truncate,
        devices=devices,
        eigenvalues=val_1,
        eigenvectors=vec_1,
        structure_tensor=S_1,
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
