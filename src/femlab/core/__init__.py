"""Core FEM modules: elements, quadrature, assembly, BCs, solver, dynamics, nonlinear."""

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
from femlab.core.kinematics import deformation_gradient_tet4, tet4_G_matrix
from femlab.core.hyperelastic import (
    lame_parameters, neo_hookean_energy, neo_hookean_pk1, neo_hookean_tangent,
)
from femlab.core.element_nl import tet4_internal_force_nl, tet4_tangent_stiffness_nl
from femlab.core.assembly_nl import (
    assemble_internal_force_tet4_nl, assemble_system_tet4_nl,
)
from femlab.core.corotational import (
    polar_decomposition_tet4, tet4_internal_force_cr, tet4_tangent_stiffness_cr,
)
from femlab.core.assembly_cr import (
    assemble_internal_force_tet4_cr, assemble_system_tet4_cr,
)
from femlab.core.newton import solve_newton, solve_newton_general
from femlab.core.dynamics_nl import (
    backward_euler_nl, newmark_beta_nl, quasi_static_nl,
    compute_strain_energy_cr,
)

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
    # Kinematics
    "deformation_gradient_tet4", "tet4_G_matrix",
    # Hyperelastic
    "lame_parameters", "neo_hookean_energy", "neo_hookean_pk1", "neo_hookean_tangent",
    # Nonlinear element
    "tet4_internal_force_nl", "tet4_tangent_stiffness_nl",
    # Nonlinear assembly
    "assemble_internal_force_tet4_nl", "assemble_system_tet4_nl",
    # Newton solver
    "solve_newton",
    "solve_newton_general",
    # Corotational
    "polar_decomposition_tet4",
    "tet4_internal_force_cr", "tet4_tangent_stiffness_cr",
    # Corotational assembly
    "assemble_internal_force_tet4_cr", "assemble_system_tet4_cr",
    # Nonlinear dynamics
    "backward_euler_nl", "newmark_beta_nl", "quasi_static_nl",
    "compute_strain_energy_cr",
]
