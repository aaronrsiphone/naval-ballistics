# Progress Update

## Completed
- Scaffolded Typer-based CLI with run, sweep, compare, plot, and export commands verified by tests.
- Implemented atmosphere, frames, gravity, and explicit Coriolis/centrifugal terms with unit coverage.

## In Progress
- Began Modified Point-Mass (MPM) implementation with a simplified integrator that includes gravity, drag, and Earth-rotation effects. Further refinement needed for yaw-of-repose and aerodynamic tables.

## Next Steps
- Extend the MPM model to support aerodynamic coefficient tables, wind models, and registry outputs.
- Design six-degree-of-freedom dynamics scaffolding to build upon the validated primitives.
