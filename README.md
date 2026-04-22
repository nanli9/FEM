# FEMLab

Educational finite element method implementation built on NVIDIA Warp.

Implements the FEM pipeline from scratch — mesh handling, shape functions,
assembly, solving, and visualization — using Warp for GPU/CPU computation.

## Installation

```bash
# Clone
git clone https://github.com/nanli9/FEM.git
cd FEM

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- **warp-lang** — compute backend (arrays, kernels, sparse ops)
- **numpy** — array utilities
- **scipy** — sparse solvers (future milestones)
- **pyvista** — mesh visualization
- **pytest** / **pytest-cov** — testing (dev only)

## Milestone 0: Environment Bootstrap

**Status**: Complete

What is implemented:
- Project skeleton (`src/femlab/` package)
- Warp init and device detection
- PyVista viewer (`femlab.viz`) for scalar and vector fields
- Sanity check script proving Warp kernels + PyVista rendering work
- Test suite for environment verification

### Run the demo

```bash
python scripts/run_case.py sanity
```

### Run tests

```bash
pytest
```

## Milestone 1: 2D Linear Static Elasticity

**Status**: Complete

What is implemented:
- T3 (3-node triangle) shape functions and gradients (`femlab.core.basis`)
- Gauss quadrature on reference triangle, orders 1 and 2 (`femlab.core.quadrature`)
- B-matrix (strain-displacement) for T3 elements (`femlab.core.element`)
- Plane stress and plane strain constitutive matrices (`femlab.core.material`)
- Element stiffness matrix, residual, and stress recovery (`femlab.core.element`)
- Global sparse assembly via COO→CSR (`femlab.core.assembly`)
- Dirichlet BC application by elimination (`femlab.core.boundary`)
- Sparse direct linear solve with diagnostics (`femlab.core.solver`)
- Structured rectangle mesh generator with T3 triangulation (`femlab.mesh.rectangle`)
- Patch tests (uniform strain + pure shear) — pass to 1e-10 tolerance
- Cantilever beam benchmark — convergence verified against Euler-Bernoulli theory

### Run the cantilever demo

```bash
python scripts/run_case.py cantilever
```

### Run tests

```bash
pytest
```

### Known limitations

- T3 (constant-strain) elements converge slowly in bending; use fine meshes for accuracy
- Assembly loop is in Python (no Warp kernel acceleration yet)
- Only structured rectangular meshes supported

## Milestone 2: 3D Linear Static / Mass Matrix & Time Stepping

**Status**: Complete

What is implemented:
- Tet4 (4-node tetrahedron) shape functions and gradients (`femlab.core.basis`)
- Gauss quadrature on reference tetrahedron, orders 1 and 2 (`femlab.core.quadrature`)
- 3D isotropic constitutive matrix (`femlab.core.material`)
- Tet4 B-matrix, element stiffness, residual, and stress recovery (`femlab.core.element`)
- 3D global sparse assembly (`femlab.core.assembly`)
- Structured box mesh generator with Freudenthal Tet4 triangulation (`femlab.mesh.box`)
- Consistent and lumped mass matrices for T3 and Tet4 (`femlab.core.mass`)
- Central difference (explicit) time integrator with lumped mass (`femlab.core.dynamics`)
- Newmark-beta (implicit, β=1/4, γ=1/2) time integrator (`femlab.core.dynamics`)
- 3D patch tests (uniaxial stretch + pure shear) — pass to 1e-9 tolerance
- Mass conservation tests (lumped vs consistent row-sum equivalence)
- Energy conservation tests for both integrators
- Cantilever natural frequency benchmark against analytical beam theory

### Run the demos

```bash
python scripts/run_case.py beam3d       # 3D static cantilever
python scripts/run_case.py vibration    # 2D dynamic free vibration
```

### Known limitations

- Tet4 (constant-strain) elements converge slowly in bending, like T3
- Assembly loops are in Python (no Warp kernel acceleration yet)
- Only structured meshes (rectangle, box) supported
- No damping in time integrators

## Milestone 3: Deformation Gradient, Hyperelastic Materials & Newton Solver

**Status**: Complete

What is implemented:
- Deformation gradient F from nodal positions (`femlab.core.kinematics`)
- G matrix (9×12) mapping DOFs to vectorized F for compact force/stiffness expressions
- Compressible Neo-Hookean hyperelastic material model (`femlab.core.hyperelastic`)
- First Piola-Kirchhoff stress P = ∂W/∂F and material tangent A = ∂P/∂F (9×9)
- Nonlinear Tet4 element internal force and tangent stiffness (`femlab.core.element_nl`)
- Nonlinear global assembly with combined K_T + f_int pass (`femlab.core.assembly_nl`)
- Newton-Raphson solver with convergence diagnostics (`femlab.core.newton`)
- K_T at u=0 verified to equal linear stiffness (nonlinear ↔ linear bridge test)
- Numerical verification of P = ∂W/∂F and A = ∂P/∂F via finite differences
- Large-deformation cantilever demo with load stepping and load-displacement curve

### Run the demo

```bash
python scripts/run_case.py cantilever_nl
```

### Known limitations

- Assembly loops are in Python (no Warp kernel acceleration yet)
- No line search or arc-length control — relies on load stepping for convergence
- Only Neo-Hookean material; no Mooney-Rivlin or Ogden yet
- Displacement-based formulation may lock for nearly incompressible materials (ν → 0.5)

## Milestone 4: Corotational FEM

**Status**: Complete

What is implemented:
- Polar decomposition of F → rotation R per element via `scipy.linalg.polar` (`femlab.core.corotational`)
- Block rotation matrix T_R = blkdiag(R,R,R,R) for transforming forces/displacements
- Corotational internal force: remove rotation → linear force → rotate back
- Corotational tangent stiffness: T_R · K_linear · T_R^T + geometric stiffness K_σ
- Geometric (initial-stress) stiffness correction from local stress state
- Corotational global assembly via COO→CSR (`femlab.core.assembly_cr`)
- General-purpose Newton solver (`solve_newton_general`) for arbitrary assembly functions
- Large-rotation tests: pure 90°/120° rotation gives zero force (linear FEM gives large spurious force)
- Rotation + stretch tests: corotational extracts correct local deformation
- Tangent consistency verified against finite differences
- Comparison demo: Linear vs Corotational vs Full Nonlinear (Neo-Hookean) load-displacement curves

### Run the demo

```bash
python scripts/run_case.py cantilever_cr
```

### Run tests

```bash
pytest
```

### Known limitations

- Assembly loops are in Python (no Warp kernel acceleration yet)
- Geometric stiffness uses the standard initial-stress formula (approximate for the corotational formulation)
- Corotational assumes small strains — for large-strain problems use the full nonlinear solver
- Only Tet4 elements supported for corotational formulation

## Milestone 5 (Next): Research — XFEM / Cutting

- [ ] Enrichment functions for discontinuities
- [ ] Level-set representation of cuts
- [ ] Modified assembly for enriched elements
