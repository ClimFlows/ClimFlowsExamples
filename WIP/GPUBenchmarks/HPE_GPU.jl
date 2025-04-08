# ## Preamble
using Revise
using Pkg;
Pkg.activate(@__DIR__);

using KernelAbstractions, Adapt, ManagedLoops, LoopManagers
using LinearAlgebra: mul!
using PrettyTables

using NetCDF: ncread, ncwrite, nccreate, ncclose
using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile

using ManagedLoops: synchronize, @with, @vec, @unroll
using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
# using SIMDMathFunctions
using MutatingOrNot: void, similar!

using CFDomains: CFDomains, Stencils, VoronoiSphere, SigmaCoordinate
using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
using CFPlanets: ShallowTradPlanet
using ClimFluids: IdealPerfectGas
using CFHydrostatics: CFHydrostatics, HPE, diagnostics
using ClimFlowsTestCases: Jablonowski06, describe, initial

Base.getindex(::ManagedLoops.DeviceManager, x::AbstractArray) = x

function setup(sphere, choices, params, cpu)
    case = choices.TestCase(choices.precision; params.testcase...)
    params = merge(choices, case.params, params)
    ## physical parameters needed to run the model
    @info case

    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = choices.coordinate(params.nz[1], params.ptop)

    surface_geopotential(lon, lat) = initial(case, lon, lat)[2]
    model = HPE(params, cpu, sphere, vcoord, surface_geopotential, gas)

    ## initial condition & standard diagnostics
    state = let
        init(lon, lat) = initial(case, lon, lat)
        init(lon, lat, p) = initial(case, lon, lat, p)
        CFHydrostatics.initial_HPE(init, model)
    end

    return model, diagnostics(model), state
end

function mintime(fun!, out_, mgr, args_)
    @info "$fun! on $mgr"
    args = args_ |> mgr
    out = out_ |> mgr
    function work()
        for _ in 1:10
            fun!(out, mgr, args...)
        end
        synchronize(mgr)
    end
    return minimum((@timed work()).time for _ in 1:10)
end

function bench(fun!, mgrs, args...)
    out = fun!(void, mgrs[1], args...)
    times = (mintime(fun!, out, mgr, args) for mgr in mgrs)
    return permutedims([fun!, times...])
end

mmul!(P, _, M, N) = mul!(similar!(P, M), M, N)
vexp!(y, mgr, x) = @. mgr[y] = @fastmath log(exp(x))

include("voronoi.jl")

choices = (
    gpu_blocks=(0, 32),
    precision=Float32,
    nz=(32, 96, 96*4),
    # numerics
    meshname = DYNAMICO_meshfile("uni.1deg.mesh.nc"),
    coordinate = SigmaCoordinate,
    consvar = :temperature,
    TimeScheme = KinnmarkGray{2,5}, # RungeKutta4,
    # physics
    Fluid = IdealPerfectGas,
    TestCase = Jablonowski06
)

params = (
    # numerics
    courant = 4.0,
    # physics
    radius = 6.4e6,
    Omega = 7.27220521664304e-5,
    ptop = 225.52395239472398, # compatible with NCARL30 vertical coordinate
    Cp = 1004.5,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    nu_gradrot = 1e-16,
    hyperdiff_nu = 0.002,
    # simulation
    testcase = (), # to override default test case parameters
    interval = 6 * 3600, # 6-hour intervals between saved snapshots
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere

gpu0, gpu = plain, simd = PlainCPU(), VectorizedCPU()

try
    global gpu, gpu0
    using oneAPI
    if oneAPI.functional()
        @info "Functional oneAPI GPU detected !"
        oneAPI.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), choices.gpu_blocks)
    end
catch e
end

try
    global gpu, gpu0
    using CUDA
    if CUDA.functional()
        @info "Functional CUDA GPU detected !"
        CUDA.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(CUDABackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(CUDABackend(), choices.gpu_blocks)
    end
catch e
end

model, diagnostics(model), state = setup(vsphere, choices, params, simd)

for nz in choices.nz
    M = randn(choices.precision, 1024, 1024)
    N = randn(choices.precision, 1024, 1024)
    ue = randn(choices.precision, nz, length(edges(vsphere)))
    qi = randn(choices.precision, nz, length(cells(vsphere)))
    qv = randn(choices.precision, nz, length(duals(vsphere)))

    data = vcat(
    bench(vexp!, mgrs, M),
    bench(mmul!, mgrs, M, N),
    bench(gradient!, mgrs, vsphere, qi),
    bench(gradperp!, mgrs, vsphere, qv),
    bench(perp!, mgrs, vsphere, ue),
    bench(curl!, mgrs, vsphere, ue),
    bench(divergence!, mgrs, vsphere, ue),
    bench(TRiSK!, mgrs, vsphere, ue))

    header = (["nz=$nz", "CPU", "SIMD", "GPU", "GPU(blocked)"])
    best = Highlighter((data, i, j) -> j>1 && all(data[i, k] >= data[i, j] for k in 2:size(data, 2)),
                   crayon"red bold")

    pretty_table(data;
             header=header,
             formatters=ft_printf("%7.6f", 2:5),
             header_crayon=crayon"yellow bold",
             highlighters=best,
             tf=tf_unicode_rounded)
end

GC.gc(true) # free GPU resources before exiting to avoid segfault ?
