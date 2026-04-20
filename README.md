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

## Milestone 1 (Next): 2D Linear Static Elasticity

- [ ] T3 shape functions and gradients
- [ ] Gauss quadrature on reference triangle
- [ ] B-matrix (strain-displacement)
- [ ] Plane stress/strain constitutive matrix
- [ ] Element stiffness matrix
- [ ] Global sparse assembly
- [ ] Dirichlet BC application
- [ ] Linear solve
- [ ] Patch test
- [ ] Cantilever beam benchmark
