using Pkg; Pkg.activate(@__DIR__)
using Revise

using NetCDF: ncread
using BenchmarkTools

# import Mooncake
# import ForwardDiff
# import DifferentiationInterface as DI
# using DifferentiationInterface: Constant as Const

using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile

using CFDomains: CFDomains, Stencils, VoronoiSphere, transpose!, void
import CFDomains.VoronoiOperators as Ops

using LoopManagers: VectorizedCPU

choices = (precision = Float32, nz=32, batch=3, meshname = "uni.1deg.mesh.nc", tol=1e-3)
reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
sphere = VoronoiSphere(reader; prec = choices.precision)
@info sphere

function test_op(mgr, op!, output, inputs...)
    op!(output, mgr, inputs...)
    display(@benchmark $op!($output, $mgr, $inputs...))
end

function test(mgr, dims...)
    @info "test" mgr dims
    on_edges() = randn(choices.precision, dims..., length(sphere.le_de)) 
    on_cells() = randn(choices.precision, dims..., length(sphere.Ai))
    on_duals() = randn(choices.precision, dims..., length(sphere.Av))

    ucov, qe, F, qv, m = on_edges(), on_edges(), on_edges(), on_duals(), on_cells();

    # test_op(mgr, Ops.Curl(sphere), qv, ucov)
    test_op(mgr, Ops.Divergence(sphere), m, ucov)
    test_op(mgr, Ops.EnergyTRiSK(sphere), ucov, qe, F)
end

simd8 = VectorizedCPU(8)
simd16 = VectorizedCPU(16)
simd32 = VectorizedCPU(32)

for mgr in (nothing, simd8)
    test(mgr, choices.nz * choices.batch)
    test(mgr, choices.nz, choices.batch)
end
