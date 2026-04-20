#!/usr/bin/env python3
"""Milestone 0 sanity check: verify Warp + PyVista environment.

Creates a triangulated grid, computes a scalar distance field via a Warp
kernel, derives a radial vector field, and renders both with PyVista.
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp
import pyvista as pv

# Ensure femlab is importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import femlab
from femlab.viz import show_scalar_field, show_vector_field


@wp.kernel
def compute_distance_field(
    points: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    out[tid] = wp.sqrt(dx * dx + dy * dy + dz * dz)


def make_grid_mesh(nx: int = 10, ny: int = 10):
    """Regular triangulated grid on [0,1]^2, z=0."""
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros((nx + 1) * (ny + 1))])

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])
    return points.astype(np.float64), np.array(cells, dtype=np.int32)


def main():
    print("=" * 60)
    print("FEMLab Milestone 0 — Environment Sanity Check")
    print("=" * 60)

    # --- environment info ---
    info = femlab.get_device_info()
    print(f"\nWarp version:    {info['warp_version']}")
    print(f"CUDA available:  {info['cuda_available']}")
    print(f"Active device:   {info['device']}")
    print(f"NumPy version:   {np.__version__}")
    print(f"PyVista version: {pv.__version__}")
    print(f"Python version:  {sys.version.split()[0]}")

    # --- mesh ---
    print("\nCreating 10x10 triangulated grid...")
    points, cells = make_grid_mesh(10, 10)
    print(f"  Vertices:  {len(points)}")
    print(f"  Triangles: {len(cells)}")

    # --- scalar field via Warp kernel ---
    print("\nComputing distance field via Warp kernel...")
    wp_pts = wp.from_numpy(points.astype(np.float32), dtype=wp.vec3)
    wp_out = wp.zeros(len(points), dtype=wp.float32)
    center = wp.vec3(0.5, 0.5, 0.0)
    wp.launch(compute_distance_field, dim=len(points), inputs=[wp_pts, center, wp_out])
    wp.synchronize()

    scalars = wp_out.numpy()
    print(f"  Min distance: {scalars.min():.4f}")
    print(f"  Max distance: {scalars.max():.4f}")

    # --- vector field (radial direction) ---
    print("\nComputing radial direction field...")
    diff = points - np.array([0.5, 0.5, 0.0])
    norms = np.linalg.norm(diff, axis=1, keepdims=True).clip(min=1e-10)
    vectors = (diff / norms).astype(np.float64)

    # --- render ---
    print("\nRendering scalar field (close window to continue)...")
    show_scalar_field(
        points, cells, scalars,
        scalar_name="Distance from center",
        title="Milestone 0: Scalar Field",
    )

    print("Rendering vector field (close window to finish)...")
    show_vector_field(
        points, cells, vectors,
        vector_name="Radial direction",
        scale=0.05,
        title="Milestone 0: Vector Field",
    )

    print("\n" + "=" * 60)
    print("Milestone 0 sanity check PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
