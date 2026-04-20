"""Visualization tests for Milestone 0."""

import numpy as np
import pytest
import pyvista as pv

from femlab.viz import show_mesh, show_scalar_field, show_vector_field


@pytest.fixture
def tri_mesh():
    """Minimal 2-triangle quad mesh."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return points, cells


def test_show_mesh(tri_mesh):
    pv.OFF_SCREEN = True
    pts, cells = tri_mesh
    pl = show_mesh(pts, cells, off_screen=True)
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_show_scalar_field(tri_mesh):
    pv.OFF_SCREEN = True
    pts, cells = tri_mesh
    scalars = np.linalg.norm(pts - 0.5, axis=1)
    pl = show_scalar_field(pts, cells, scalars, off_screen=True)
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_show_vector_field(tri_mesh):
    pv.OFF_SCREEN = True
    pts, cells = tri_mesh
    vecs = np.column_stack([np.ones(4), np.zeros(4), np.zeros(4)])
    pl = show_vector_field(pts, cells, vecs, off_screen=True)
    assert isinstance(pl, pv.Plotter)
    pl.close()
