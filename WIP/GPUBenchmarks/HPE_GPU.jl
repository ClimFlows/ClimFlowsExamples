# ## Preamble
using Revise
using Pkg;
Pkg.activate(@__DIR__);

using CUDA, oneAPI, KernelAbstractions, Adapt, ManagedLoops, LoopManagers
using LinearAlgebra: mul!
using PrettyTables

using NetCDF: ncread, ncwrite, nccreate, ncclose
using ClimFlowsData: DYNAMICO_reader

using ManagedLoops: synchronize, @with, @vec, @unroll
using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
using SIMDMathFunctions

using CFDomains: CFDomains, Stencils, VoronoiSphere
using MutatingOrNot: void, similar!

Base.getindex(::ManagedLoops.DeviceManager, x::AbstractArray) = x

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

choices = (gpu_blocks=(0, 32),
           precision=Float32,
           meshname="uni.1deg.mesh.nc",
           nz=96)
reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)
           
if oneAPI.functional()
    @info "Functional oneAPI GPU detected !"
    oneAPI.versioninfo()
    gpu0 = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), (0, 0))
    gpu = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), choices.gpu_blocks)
elseif CUDA.functional()
    @info "Functional CUDA GPU detected !"
    CUDA.versioninfo()
    gpu0 = LoopManagers.KernelAbstractions_GPU(CUDABackend(), (0, 0))
    gpu = LoopManagers.KernelAbstractions_GPU(CUDABackend(), choices.gpu_blocks)
end

plain = PlainCPU()
simd = VectorizedCPU()
mgrs = (plain, simd, gpu0, gpu)

M = randn(choices.precision, 1024, 1024)
N = randn(choices.precision, 1024, 1024)
ue = randn(choices.precision, choices.nz, length(edges(vsphere)))
qi = randn(choices.precision, choices.nz, length(cells(vsphere)))
qv = randn(choices.precision, choices.nz, length(duals(vsphere)))

data = vcat(
    bench(vexp!, mgrs, M),
    bench(mmul!, mgrs, M, N),
    bench(gradient!, mgrs, vsphere, qi),
    bench(gradperp!, mgrs, vsphere, qv),
    bench(perp!, mgrs, vsphere, ue),
    bench(curl!, mgrs, vsphere, ue),
    bench(divergence!, mgrs, vsphere, ue),
    bench(TRiSK!, mgrs, vsphere, ue))

header = (["Function", "Plain", "SIMD", "GPU", "GPU (blocked)"],)
best = Highlighter((data, i, j) -> j>1 && all(data[i, k] >= data[i, j] for k in 2:size(data, 2)),
                   crayon"red bold")

pretty_table(data;
             header=header,
             formatters=ft_printf("%7.6f", 2:5),
             header_crayon=crayon"yellow bold",
             highlighters=best,
             tf=tf_unicode_rounded)

GC.gc(true) # free GPU resources before exiting to avoid segfault ?
