"""Mesh and topology utilities."""

from femlab.mesh.rectangle import rectangle_mesh, boundary_nodes
from femlab.mesh.box import box_mesh, box_boundary_nodes

__all__ = [
    "rectangle_mesh", "boundary_nodes",
    "box_mesh", "box_boundary_nodes",
]
