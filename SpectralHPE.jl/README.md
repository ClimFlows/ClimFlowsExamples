# Hydrostatic compressible equations, spectral harmonics

This example solves the hydrostatic compressible equations equations on the sphere.
The traditional, shallow-atmosphere, spherical-geoid approximations are made, with terrestrial parameters. Thermodynamics are those of an ideal gas.
The initial condition is zonally-symmetric with a perturbation that develops into a baroclinic instability (Jablonowsky & Williamson, 2006).
The movie displays temperature on the 850 hPa isobar.

Time integration is fully expicit and uses a Runge-Kutta scheme (Kinnmark and Gray, 1984).
The vertical coordinate is Lagrangian. For long-term stability, prognostic fields are 
vertically remapped to a hybrid pressure coordinate after each full Runge-Kutta scheme. 
Mass budget is solved in flux-form and momentum budget in curl form (vector-invariant form). 

Spectral harmonics computations use [SHTnsSpheres](https://github.com/ClimFlows/SHTnsSpheres.jl), 
which wraps the [SHTns library](https://nschaeff.bitbucket.io/shtns/) by Nathanael Schaeffer.

https://github.com/ClimFlows/.github/assets/24214175/4410dfe0-eff4-4b8c-b17b-546103ba6579

