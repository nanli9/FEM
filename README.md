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

## Milestone 2 (Next): 3D Linear Static / Mass Matrix & Time Stepping

- [ ] Tet4 shape functions and gradients
- [ ] 3D constitutive matrix
- [ ] 3D element stiffness and assembly
- [ ] Consistent/lumped mass matrix
- [ ] Time integration (Newmark or central difference)
