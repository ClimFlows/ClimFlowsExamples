# Hydrostatic compressible equations, spectral harmonics

This example solves the hydrostatic compressible equations equations on the sphere.
The traditional, shallow-atmosphere, spherical-geoid approximations are made, with terrestrial parameters. Thermodynamics are those of an ideal gas.
The initial condition is zonally-symmetric with a perturbation that develops into a baroclinic instability (Jablonowski & Williamson, 2006).

Time integration is fully explicit and uses a Runge-Kutta scheme (Kinnmark and Gray, 1984).
The vertical coordinate is Lagrangian. For long-term stability, prognostic fields are 
vertically remapped to a hybrid pressure coordinate after each full Runge-Kutta scheme. 
Mass budget is solved in flux-form and momentum budget in curl form (vector-invariant form). 

Spectral harmonics computations use [SHTnsSpheres](https://github.com/ClimFlows/SHTnsSpheres.jl), 
which wraps the [SHTns library](https://nschaeff.bitbucket.io/shtns/) by Nathanael Schaeffer.

This example can use several CPU cores (threads). To use two threads:
```shell
export JULIA_NUM_THREADS=2
export JULIA_EXCLUSIVE=true
julia --startup-file=no SpectralHPE.jl
```

https://github.com/user-attachments/assets/71d3168e-2d8e-4c71-9cf8-9b0d936a8881

