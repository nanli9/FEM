"""Core FEM modules: elements, quadrature, assembly, BCs, solver, dynamics."""

from femlab.core.basis import (
    t3_shape, t3_grad_ref, t3_grad_phys, t3_jacobian,
    tet4_shape, tet4_grad_ref, tet4_grad_phys, tet4_jacobian,
)
from femlab.core.quadrature import triangle_quadrature, tetrahedron_quadrature
from femlab.core.material import plane_stress_D, plane_strain_D, isotropic_3d_D
from femlab.core.element import (
    t3_B_matrix, t3_element_stiffness, t3_element_residual, t3_stress,
    tet4_B_matrix, tet4_element_stiffness, tet4_element_residual, tet4_stress,
)
from femlab.core.assembly import (
    assemble_global_stiffness, assemble_global_force,
    assemble_global_stiffness_tet4, assemble_global_force_tet4,
)
from femlab.core.boundary import apply_dirichlet
from femlab.core.solver import solve_linear
from femlab.core.mass import (
    assemble_global_mass_t3, assemble_global_mass_tet4,
)
from femlab.core.dynamics import central_difference, newmark_beta

__all__ = [
    # T3 basis
    "t3_shape", "t3_grad_ref", "t3_grad_phys", "t3_jacobian",
    # Tet4 basis
    "tet4_shape", "tet4_grad_ref", "tet4_grad_phys", "tet4_jacobian",
    # Quadrature
    "triangle_quadrature", "tetrahedron_quadrature",
    # Material
    "plane_stress_D", "plane_strain_D", "isotropic_3d_D",
    # T3 element
    "t3_B_matrix", "t3_element_stiffness", "t3_element_residual", "t3_stress",
    # Tet4 element
    "tet4_B_matrix", "tet4_element_stiffness", "tet4_element_residual", "tet4_stress",
    # Assembly
    "assemble_global_stiffness", "assemble_global_force",
    "assemble_global_stiffness_tet4", "assemble_global_force_tet4",
    # BCs & solver
    "apply_dirichlet", "solve_linear",
    # Mass
    "assemble_global_mass_t3", "assemble_global_mass_tet4",
    # Dynamics
    "central_difference", "newmark_beta",
]
