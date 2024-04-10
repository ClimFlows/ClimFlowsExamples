# Weno

Weno implements WENO reconstructions on rectangular arrays. It provides a function to compute div(U q), with U a velocity and q a tracer field. It assumes a C-grid staggering.


## Example

```
# make sure your current directory is 'WIP/Weno.jl'
pwd() |> display

# install dependencies
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# advection of a passive tracer in a closed square box
include("examples/adv_in_a_box.jl")
```
