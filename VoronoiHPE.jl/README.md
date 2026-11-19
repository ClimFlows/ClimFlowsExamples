# Hydrostatic compressible equations, mimetic finite difference / finite volume on Voronoi mesh

This example solves the hydrostatic compressible equations equations on the sphere.
The traditional, shallow-atmosphere, spherical-geoid approximations are made, with terrestrial parameters. Thermodynamics are those of an ideal gas.
The initial condition is zonally-symmetric with a perturbation that develops into a baroclinic instability (Jablonowski & Williamson, 2006).

Time integration is fully explicit and uses a Runge-Kutta scheme (Kinnmark and Gray, 1984).
The vertical coordinate is Lagrangian. For long-term stability, prognostic fields are 
vertically remapped to a hybrid pressure coordinate after each full Runge-Kutta scheme. 
Mass budget is solved in flux-form and momentum budget in curl (vector-invariant) form, (Dubos et al. GMD 2015). 

## Run on CPU or CUDA

Run these commands from `VoronoiHPE.jl/` with Julia 1.11 or later. Both
environments run the same case and write `VoronoiHPE.nc` in the working
directory. Move an existing output file before rerunning the example.

### CPU (Float64)

```sh
julia --project=CPU install.jl
julia --project=CPU --threads=2 --startup-file=no VoronoiHPE.jl
```

Running `julia install.jl` or `julia VoronoiHPE.jl` also selects the CPU
environment by default.

### CUDA (Float32)

Use an NVIDIA GPU with a driver that supports CUDA 12.9. The CUDA project
selects that runtime.

```sh
julia --project=CUDA install.jl
julia --project=CUDA --startup-file=no VoronoiHPE.jl
```

The initial condition and Voronoi mesh are prepared on the CPU. The model and
state are then adapted to CUDA for time integration. Snapshots are copied back
to the CPU for interpolation and NetCDF output.

Choices and parameters are defined in `config.jl` and can be modified, to some
extent.
