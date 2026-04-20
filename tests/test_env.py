"""Environment sanity tests for Milestone 0."""

import numpy as np
import warp as wp


def test_warp_import():
    assert hasattr(wp, "__version__")


def test_numpy_import():
    assert hasattr(np, "__version__")


def test_pyvista_import():
    import pyvista as pv
    assert hasattr(pv, "__version__")


def test_femlab_import():
    import femlab
    info = femlab.get_device_info()
    assert "warp_version" in info
    assert "device" in info


def test_warp_array_roundtrip():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    wp_arr = wp.from_numpy(data, dtype=wp.float32)
    result = wp_arr.numpy()
    np.testing.assert_allclose(result, data)


def test_warp_kernel_launch():
    @wp.kernel
    def double_it(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        y[tid] = x[tid] * 2.0

    n = 4
    x = wp.from_numpy(np.arange(n, dtype=np.float32))
    y = wp.zeros(n, dtype=wp.float32)
    wp.launch(double_it, dim=n, inputs=[x, y])
    wp.synchronize()
    np.testing.assert_allclose(y.numpy(), np.arange(n) * 2.0)
