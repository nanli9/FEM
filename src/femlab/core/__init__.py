"""Core FEM modules: elements, quadrature, assembly, BCs, solver."""

from femlab.core.basis import t3_shape, t3_grad_ref, t3_grad_phys, t3_jacobian
from femlab.core.quadrature import triangle_quadrature
from femlab.core.material import plane_stress_D, plane_strain_D
from femlab.core.element import (
    t3_B_matrix,
    t3_element_stiffness,
    t3_element_residual,
    t3_stress,
)
from femlab.core.assembly import assemble_global_stiffness, assemble_global_force
from femlab.core.boundary import apply_dirichlet
from femlab.core.solver import solve_linear

__all__ = [
    "t3_shape",
    "t3_grad_ref",
    "t3_grad_phys",
    "t3_jacobian",
    "triangle_quadrature",
    "plane_stress_D",
    "plane_strain_D",
    "t3_B_matrix",
    "t3_element_stiffness",
    "t3_element_residual",
    "t3_stress",
    "assemble_global_stiffness",
    "assemble_global_force",
    "apply_dirichlet",
    "solve_linear",
]
