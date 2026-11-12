# Hydrostatic compressible equations, spectral harmonics

This example solves the hydrostatic compressible equations on the sphere.
The traditional, shallow-atmosphere, spherical-geoid approximations are made, with terrestrial parameters. Thermodynamics are those of an ideal gas.
The initial condition is zonally-symmetric with a perturbation that develops into a baroclinic instability (Jablonowski & Williamson, 2006).

Time integration is fully explicit and uses a Runge-Kutta scheme (Kinnmark and Gray, 1984).
The vertical coordinate is Lagrangian. For long-term stability, prognostic fields are 
vertically remapped to a hybrid pressure coordinate after each full Runge-Kutta scheme. 
Mass budget is solved in flux-form and momentum budget in curl form (vector-invariant form). 

Spectral harmonics computations use [SHTnsSpheres](https://github.com/ClimFlows/SHTnsSpheres.jl), 
which wraps the [SHTns library](https://nschaeff.bitbucket.io/shtns/) by Nathanael Schaeffer.

## Run on CPU or CUDA

Run these commands from `SpectralHPE.jl/` with Julia 1.11 or later. Both
environments run the same baroclinic instability case and write
`SpectralHPE.nc` in the working directory. Move an existing output file
before rerunning the example.

### CPU (Float64)

```sh
julia --project=CPU install.jl
julia --project=CPU --threads=2 --startup-file=no SpectralHPE.jl
```

Running `julia install.jl` or `julia SpectralHPE.jl` also selects the CPU
environment by default.

### CUDA (Float32)

Use an NVIDIA GPU with a driver that supports CUDA 12.9. The CUDA project
selects that runtime to match the SHTns_jll 3.7 CUDA artifact.

```sh
julia --project=CUDA install.jl
julia --project=CUDA --startup-file=no SpectralHPE.jl
```

The analytical initial condition is sampled on the CPU. The solver advances
`Float32` grid fields and `ComplexF32` spectral coefficients on the GPU.
The CUDA sphere prepares SHTns plans for surface, layer and interface
fields once and reuses them throughout the run. Fields are copied to the
CPU for display, pressure-level interpolation and NetCDF output.

https://github.com/user-attachments/assets/71d3168e-2d8e-4c71-9cf8-9b0d936a8881
