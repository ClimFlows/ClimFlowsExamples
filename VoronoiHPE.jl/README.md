# Hydrostatic compressible equations, mimetic finite difference / finite volume on Voronoi mesh

This example solves the hydrostatic compressible equations equations on the sphere.
The traditional, shallow-atmosphere, spherical-geoid approximations are made, with terrestrial parameters. Thermodynamics are those of an ideal gas.
The initial condition is zonally-symmetric with a perturbation that develops into a baroclinic instability (Jablonowski & Williamson, 2006).

Time integration is fully explicit and uses a Runge-Kutta scheme (Kinnmark and Gray, 1984).
The vertical coordinate is Lagrangian. For long-term stability, prognostic fields are 
vertically remapped to a hybrid pressure coordinate after each full Runge-Kutta scheme. 
Mass budget is solved in flux-form and momentum budget in curl (vector-invariant) form, (Dubos et al. GMD 2015). 

Install dependencies with:
```shell
julia install.jl
```

Choices and parameters are defined in `config.jl` and can be modified, to some extent.

Run with :
```shell
julia --startup-file=no VoronoiHPE.jl
```

If detected, a GPU recognized by `CUDA` or `oneAPI` will be used. The simulation produces a NetCDF file `VoronoiHPE.nc`.
