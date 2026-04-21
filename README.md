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

## Milestone 3 (Next): Corotational FEM

- [ ] Rotation extraction per element
- [ ] Corotational element stiffness and residual
- [ ] Newton iteration for geometric nonlinearity
- [ ] Large-deformation cantilever benchmark
