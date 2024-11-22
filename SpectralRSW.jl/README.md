# Rotating Shallow-Water equations, spectral harmonics

This example solves the shallow-water equations on the sphere. 
Spectral harmonics computations use [SHTnsSpheres](https://github.com/ClimFlows/SHTnsSpheres.jl), 
which wraps the [SHTns library](https://nschaeff.bitbucket.io/shtns/) by Nathanael Schaeffer.

Mass budget is solved in flux-form and momentum budget in curl form (vector-invariant form). 
Time integration is fully expicit and uses a Runge-Kutta scheme.
The initial condition is a Rossby-Haurwitz wave (Williamson, 1991). 

The movie displays potential vorticity.

https://github.com/ClimFlows/.github/assets/24214175/4410dfe0-eff4-4b8c-b17b-546103ba6579

