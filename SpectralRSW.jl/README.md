# Rotating Shallow-Water equations, spectral harmonics

This example solves the shallow-water equations on the sphere. 
Spectral harmonics computations use [SHTnsSpheres](https://github.com/ClimFlows/SHTnsSpheres.jl), 
which wraps the [SHTns library](https://nschaeff.bitbucket.io/shtns/) by Nathanael Schaeffer.

Mass budget is solved in flux-form and momentum budget in curl form (vector-invariant form). 
Time integration is fully explicit and uses a Runge-Kutta scheme.
The initial condition is a Rossby-Haurwitz wave (Williamson, 1991). 

The movie displays potential vorticity.

https://github.com/ClimFlows/.github/assets/24214175/4410dfe0-eff4-4b8c-b17b-546103ba6579

## Run on CPU or CUDA

Run these commands from `SpectralRSW.jl/`. Both environments use the same
simulation and produce `PV.mp4` in this directory.

### CPU (Float64)

The CPU environment was verified with Julia 1.10.

```sh
julia --project=CPU install.jl
julia --project=CPU SpectralRSW.jl
```

Running `julia install.jl` and `julia SpectralRSW.jl` still selects the CPU
environment by default.

### CUDA (Float32)

Use Julia 1.12 and an NVIDIA GPU with a driver that supports CUDA 12.9.
The CUDA project selects the CUDA 12.9 runtime so SHTns_jll 3.7 loads
its CUDA artifact, including on drivers that support CUDA 13.

```sh
julia --project=CUDA install.jl
julia --project=CUDA SpectralRSW.jl
```

The analytical initial condition is sampled on the CPU. The solver advances
`Float32` grid fields and `ComplexF32` spectral coefficients on the GPU.
Potential-vorticity frames are copied back to the CPU for plotting.
