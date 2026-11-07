using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

##

using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
using NetCDF: ncread, ncwrite, nccreate, ncclose

using Adapt: Adapt, adapt
using KernelAbstractions
using CUDA: CUDA, CUDABackend, CuArray, CuDevice, device!, i32, @cuda, blockIdx, blockDim, threadIdx, launch_configuration
CUDA_functional = CUDA.functional()

using LoopManagers: PlainCPU, MultiThread, VectorizedCPU # CPU
using LoopManagers: KernelAbstractions_GPU as KA_GPU, GPUConfig, SRange, I32 # GPU
using ManagedLoops: ManagedLoops, @with, @loops, @vec, @unroll, synchronize, configure, LoopManager, DeviceManager
using CFDomains: CFDomains, VoronoiSphere
using CFDomains: VoronoiOperators as Ops

using Adapt
Adapt.@adapt_structure Ops.EnergyTRiSK

include("baseline.jl")

struct One end
@inline Base.:+(x::UInt32, ::One) = x+one(x)
@inline Base.:-(x::UInt32, ::One) = x-one(x)
const uno = One()

struct Config{Warpsize, NWarp, Repeat} 
    Config(a,b,c) = new{Int(a), Int(b), Int(c)}()
end

# make_gpu(backend, ::Config{Warpsize, NWarp}) where {Warpsize, NWarp} = KA_GPU(backend, (0,Int(Warpsize*NWarp)))

# Block size = Warpsize * NWarp
# Each Warp deals with Repeat columns
# Each Block deals with NWarp * Repeat columns
ManagedLoops.configure(mgr, config::Config) = (mgr, config)

function ManagedLoops.offload(fun::Fun, (mgr, config)::Tuple{Any, Config}, (irange, jrange)::Tuple{A,B}, args...) where {Fun, A, B}
    offload_2D(fun, mgr.gpu, config, irange, jrange, args)
end

# called from CPU => Int
@inline blocksize(::Config{M,N}) where {M,N} = Int(M*N)
@inline divide(n::Int, ::Config{M,N,P}) where {M,N,P} = div(n, N*P)
# called from GPU => UInt32
@inline div_rem(n::UInt32, v::Val{N}) where N = n >> logtwo(v), n & I32(N-1)
@inline muladd32(a::UInt32, v::Val, b::UInt32) = (a << logtwo(v))+b

logtwo(::Val{1}) = I32(0)
logtwo(::Val{2}) = I32(1)
logtwo(::Val{4}) = I32(2)
logtwo(::Val{8}) = I32(3)
logtwo(::Val{N}) where N = logtwo(Val(div(N,16)))+UInt32(4)

#==================================================================#

function offload_2D_(fun::Fun, gpu, config::Config{W,B,R}, irange, jrange, args) where {Fun, W, B, R}
    chunk = B*R
    n = div(length(jrange)+(chunk-1), chunk) # split jrange into n chunks of size B*R
    kernel = kernel_2D_new(gpu, (W, B), (W, B, n))
    i0, j0 = first(irange), first(jrange)
    ilast, jlast = last(irange), last(jrange)
    kernel(fun, config, I32(i0), I32(ilast), I32(j0), I32(jlast), args; ndrange = (W, B, n))
    return nothing
end

@inline zero_based(x) = I32(x)-I32(1)

@kernel unsafe_indices=true function kernel_2D_new(fun, ::Config{W,B,R}, i0::UInt32, ilast::UInt32, j0::UInt32, jlast::UInt32, args) where {W,B,R}
    # i0, j0, ilast, jlast are 1-based
    i, j = @index(Local, NTuple)
    _, _, k = @index(Group, NTuple)
    i = zero_based(i) # i ∈ [0 .. W[
    j = zero_based(j) # j ∈ [0 .. B[
    k = zero_based(k) # k ∈ [0 .. Nj/(B*R)[
    
    irange = SRange(i+i0, W, ilast)
    jstart = muladd32(k, Val(B*R), j+j0)
    if R>1
        jstop = min(jlast, jstart+I32(B*(R-1)))
        while jstart <= jstop
            @inline fun((irange, (jstart,)), args...)
            jstart += I32(B)
        end
    else
        @inline fun((irange, (jstart,)), args...)
    end
#    (i,i0,j,j0,jstart,jstop,jlast,k)::NTuple{N,UInt32} where N
end

#==================================================================#

function offload_2D(fun::Fun, gpu, config, irange, jrange, args) where Fun
    block = blocksize(config)
    N = divide(length(jrange), config)
    kernel = kernel_2D(gpu, block, (block,N))
    i0, j0 = first(irange), first(jrange)
    kernel(fun, config, I32(i0), I32(last(irange)), I32(j0), I32(last(jrange)), args; ndrange = (block, N))
    return nothing
end

@kernel function kernel_2D(fun, ::Config{W,B,R}, i0::UInt32, ilast::UInt32, j0::UInt32, jlast::UInt32, args) where {W,B,R}
    # i0, j0, ilast, jlast are 1-based
    # k indexes chunks of B*R columns
    i, k = @index(Global, NTuple)
    i = zero_based(i)
    k = zero_based(k)
    chunk = B*R
    if chunk==1
        istart = i+i0 # 1-based
        jstart = k+j0 # 1-based
        istart::UInt32
        jstart::UInt32
        ranges = SRange(istart, W, ilast), (jstart,)
        @inline fun(ranges, args...)
    else
        # divide block of size B*W, indexed by i, into B warps of size W: i = i1 + j1*W
        j1, i1 = div_rem(i, Val(W))  # 0-based
        # reconstruct jrange: 
        # each warp takes care of R columns : jstart, jstart+B, ..., jstart+(R-1)*B
        jstart = muladd32(k, Val(chunk), j1+j0) # 1-based
        istart = i1+i0 # 1-based
        jstart::UInt32
        istart::UInt32
        irange = SRange(istart, W, ilast)
        jlast = min(jstart+(R-1)*B, jlast)
        while jstart <= jlast
            @inline fun((irange, (jstart,)), args...)
            jstart += B
        end
    end
end

#================================================================#

function extract(degree, indices, weights)
    deg = maximum(degree)
    for ij in eachindex(degree)
        last = indices[degree[ij], ij]
        for k in degree[ij]+1:deg
            indices[k, ij] = last
            weights[k, ij] = 0
        end
    end
    return map((ij,w)->(; ij,w), indices, weights)
end

@inline function compute(mgr, uperp, wee, ucov, qe)
    @with mgr,
    let (krange, ijrange) = axes(uperp)
        @inbounds for ij in ijrange
            w = @unroll (wee[n,ij] for n in 1:10)
            for k in krange
                qij = qe[k, ij]
                uperp[k, ij] = (@unroll sum(w[n].w*ucov[k, w[n].ij]*(qij+qe[k,w[n].ij]) for n=1:10))/2
            end
        end
    end
    return
end

@inline function compute2(mgr, config, uperp, wee, ucov, qe)
    @with configure(mgr, config),
    let (krange, ijrange) = axes(uperp)
        @inbounds for ij in ijrange
            w = @unroll (wee[n,ij] for n in 1:10)
            for k in krange
                qij = qe[k, ij]
                uperp[k, ij] = (@unroll sum(w[n].w*ucov[k, w[n].ij]*(qij+qe[k,w[n].ij]) for n=1:10))/2
            end
        end
    end
    return
end

function test_gpu(mgr, uperp, wee, ucov, qe)
    uperp = uperp |> gpu
    wee = wee |> gpu
    ucov = ucov |> gpu
    qe = qe |> gpu
    compute(mgr, uperp, wee, ucov, qe)
end

function test_gpu2(gpu, config, uperp, wee, ucov, qe, N=1)
    uperp = uperp |> gpu
    wee = wee |> gpu
    ucov = ucov |> gpu
    qe = qe |> gpu
    uperp .= 1.2345f0
    compute2(gpu, config, uperp, wee, ucov, qe)
    untouched = sum(x->x==1.2345f0, uperp)
    @info config untouched

    @time CUDA.@sync for _ in 1:N-1
        compute2(gpu, config, uperp, wee, ucov, qe)
    end
end

#==================================================#

@inline function grad!(mgr, config, ucov, left_right, qi)
    @with configure(mgr, config),
    let (krange, ijrange) = axes(ucov)
        @inbounds for ij in ijrange
            left, right = left_right[ij]
            for k in krange
                ucov[k, ij] = qi[k,left]-qi[k,right]
            end
        end
    end
    return
end

function test_grad(gpu, config, N, ucov, left_right, qi)
    @info config
    CUDA.@sync grad!(gpu, config, ucov, left_right, qi)
    @time CUDA.@sync for _ in 1:N-1
        grad!(gpu, config, ucov, left_right, qi)
    end
end

#==================================================#

@inline function memcp!(y, mgr, x)
    @with mgr,
    let irange = eachindex(y)
        @inbounds for i in irange
            i::UInt32
            y[i] = x[i]
        end
    end
    return
end

@inline function memcp!(y, mgr, x, ::Nothing)
    @with mgr,
    let (krange, ijrange) = axes(y)
        @inbounds for k in krange, ij in ijrange
            y[k,ij] = x[k,ij]
        end
    end
    return
end

@inline function memcp!(y, mgr, x, config)
    nk, nij = Int32(size(x,1)), Int32(size(x,2))
    iter = outer_iteration(nij, config)
    @with mgr,
    let (wrange, hrange) = iter
        hrange::Tuple{Int32}
        wrange::SRange
        krange, ijrange = inner_iteration(nk, nij, wrange, hrange, config)
        @inbounds for k in krange, ij in ijrange
            y[k,ij] = x[k,ij]
        end
    end
    return
end

function measure(fun, args, M=10, N=10)
    fun(args...)
    @info "success $fun" args = map(typeof, args)
    for _ in 1:M
        @time CUDA.@sync for _ in 1:N
            fun(args...)
        end
    end
end

#=================================================#

choices = (
    precision=Float32,
    nz = 96,
    meshname = DYNAMICO_meshfile("uni.1deg.mesh.nc"),
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere

left_right = [(vsphere.edge_left_right[1,ij], vsphere.edge_left_right[2,ij]) for ij in axes(vsphere.edge_left_right,2)]
wee = extract(vsphere.trisk_deg, vsphere.trisk, vsphere.wee)
ucov = randn(choices.precision, choices.nz, size(wee, 2))
qe = randn(choices.precision, choices.nz, size(wee, 2))
qi = randn(choices.precision, choices.nz, length(vsphere.Ai))
uperp = similar(ucov)

@time compute(nothing, uperp, wee, ucov, qe)

# backend = KernelAbstractions.CPU()
backend = CUDABackend()
gpu = KA_GPU(backend)

test_gpu(gpu, uperp, wee, ucov, qe)

test_gpu2(gpu, GPUConfig(0,0), uperp, wee, ucov, qe)
for (B,R) in ((0,0), (1,1), (1,16), (2,16), (4,8), (4,16))
    test_gpu2(gpu, GPUConfig(B,R), uperp, wee, ucov, qe)
    (B*R>0) && test_gpu2(gpu, Config(32, B,R), uperp, wee, ucov, qe)
end

let uperp = uperp |>gpu, ucov = ucov |> gpu, qe = qe |> gpu,
    qi = qi |> gpu, left_right = left_right |> gpu
    N=0
    for (B,R) in ((0,0), (1,1), (1,16), (1,64), (2,1), (2,8), (2,32), (4,1), (4,8), (8,1), (8,8))
        test_grad(gpu, GPUConfig(B,R), N, ucov, left_right, qi)
        (B*R>0) && test_grad(gpu, Config(32,B,R), N, ucov, left_right, qi)
    end
    #=
    trisk! = Ops.EnergyTRiSK(vsphere |> gpu, Ops.set!)
    CUDA.@sync trisk!(uperp, gpu, ucov, qe)
    for _ in 1:10
        @time CUDA.@sync trisk!(uperp, gpu, ucov, qe)
    end
    =#
end

#=
x = randn(128, 100) |> gpu ;
y = similar(x)

memcp!(y, gpu, x)
memcp!(y, gpu, x, nothing)
measure(memcp!, (y, gpu, x, Config(32,1,1)))
=#
