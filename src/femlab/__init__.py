"""FEMLab: Educational FEM on NVIDIA Warp."""

__version__ = "0.1.0"

import warp as wp

wp.init()


def get_device_info() -> dict:
    """Return a summary of the compute environment."""
    return {
        "warp_version": wp.__version__,
        "cuda_available": wp.is_cuda_available(),
        "device": str(wp.get_device()),
    }
