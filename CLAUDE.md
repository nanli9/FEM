# FEM Lab Rules

This repository is a learning-and-research FEM codebase built on Warp as a compute backend.
The goal is NOT to hide behind existing high-level FEM APIs.
The goal IS to understand and implement the FEM pipeline ourselves:
- mesh and topology
- shape functions and quadrature
- strain/stress computation
- element residuals and tangent stiffness
- global sparse assembly
- boundary condition application
- linear and nonlinear solves
- time integration
- visualization and regression tests

## Core priorities

1. Preserve a clear educational pipeline.
2. Prefer explicit math and readable implementation over clever abstractions.
3. Every new physics feature must come with a minimal benchmark and regression test.
4. Never jump ahead to XFEM, contact, or advanced features before the current milestone is validated.
5. Use Warp as a backend for arrays, kernels, sparse ops, and performance, but do not rely on `warp.fem.integrate()` for the main production solver unless explicitly asked.
6. `warp.fem` examples may be used only as:
   - installation sanity checks
   - validation references
   - debugging/comparison oracles

## Development order

Implement features in this order unless the user explicitly overrides:
1. Environment setup and visualization
2. 2D/3D linear static FEM
3. Mass matrix and time stepping
4. Nonlinear hyperelastic FEM with Newton solve (deformation gradient, Neo-Hookean, Newton-Raphson)
5. Corotational FEM (polar decomposition of F, corotational wrapper around linear elements)
6. Research branch for XFEM / cutting

## Required engineering standards

- Keep modules small and single-purpose.
- Avoid giant god-files.
- Separate math/reference code from Warp execution code.
- Every element type must have:
  - shape function implementation
  - gradient/Jacobian implementation
  - element residual
  - element stiffness
  - unit tests
- Every solver must expose:
  - inputs
  - convergence criteria
  - iteration logs
  - failure mode

## Testing requirements

Do not claim a milestone is complete unless these exist:
- unit tests for basis and quadrature
- at least one patch test
- at least one end-to-end benchmark case
- visual check script
- numerical regression tolerance documented in test comments

## Boundary conditions

Prefer explicit elimination or well-documented penalty methods.
Do not silently bake BC behavior into unrelated code.
Any BC application must be traceable in one module.

## Numerical debugging rules

When debugging:
1. first verify shape functions and quadrature
2. then verify Jacobians / element orientation
3. then verify local residual/stiffness
4. then verify scatter/assembly
5. then verify BC handling
6. then verify solver settings

Never diagnose convergence by guessing.

## Performance rules

- First make it correct on tiny cases.
- Then profile.
- Then optimize hotspots.
- Never sacrifice debuggability in the reference path.
- Keep a CPU-friendly reference path when possible.
- GPU optimization should not erase mathematical clarity.

## Visualization

Use PyVista-based tooling for quick inspection:
- undeformed mesh
- deformed mesh
- displacement magnitude
- strain/stress scalar fields
- per-element debug overlays when needed

## Code change policy

When editing code:
- make the smallest coherent change that completes the task
- do not refactor unrelated modules
- do not rename files or APIs unless necessary
- if architecture must change, explain why in comments or commit message

## Milestone completion checklist

A milestone is complete only if:
- code runs from a documented command
- tests pass
- a demo case is viewable
- logs show expected convergence/stability behavior
- README notes what is implemented and what is still missing

## Forbidden shortcuts

Do not:
- skip tests because the output “looks fine”
- introduce XFEM before linear/corotational/nonlinear basics are validated
- mix reference math and rendering hacks into the same module
- hide state mutation across unrelated files
- add dependencies unless they materially improve the project
