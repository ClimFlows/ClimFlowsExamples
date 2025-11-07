using Pkg; Pkg.activate(@__DIR__); Pkg.precompile()
using Revise
using InteractiveUtils
using BenchmarkTools
using Cthulhu

@time_imports begin
    import DifferentiationInterface as DI
    using DifferentiationInterface: Constant as Const
    import Mooncake
    import Enzyme
        
    using FixedSizeArrays
    using LinearAlgebra: dot, norm

    using ClimFlowsData: DYNAMICO_meshfile, DYNAMICO_reader
    using NetCDF: ncread
    using CFDomains: VoronoiSphere, Stencils
    import CFDomains.VoronoiOperators as Ops
end

#=============================================================================#

includet("stencils.jl")

# `choices` is for discrete parameters, while `params` is for continuous parameters (floats)
# Values in `params` will be converted to `choices.precision`.

choices = (
    precision = Float32,
    meshname = DYNAMICO_meshfile("uni.1deg.mesh.nc"),
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)

#=============== Gradient ================#

q = randn(choices.precision, length(vsphere.lon_i))
tmp = similar(q, length(vsphere.lon_e)) # gradient is computed on edges
test_op(q, tmp, Ops.Gradient(vsphere))

exit()

#=============== TRiSK ================#

q = randn(choices.precision, length(vsphere.lon_e))
tmp = similar(q)
test_op(q, tmp, Ops.TRiSK(vsphere))

#=============================================================================#

includet("stubs.jl")
includet("fixed_size.jl")
includet("rrules.jl")

include("check_fixed_size.jl")
