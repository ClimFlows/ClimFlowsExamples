# install dependencies
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using InteractiveUtils

using CUDA: CUDA, CUDABackend, CuArray, CuDevice, device!, i32, @cuda
using oneAPI: oneAPI, oneAPIBackend, oneArray

using Adapt: Adapt, adapt
using KernelAbstractions

using LoopManagers: PlainCPU, MultiThread, VectorizedCPU, KernelAbstractions_GPU
using ManagedLoops: ManagedLoops, @with, @loops, @vec, synchronize, LoopManager, DeviceManager

using GPUArrays: gpu_call, @linearidx, assume
using LinearAlgebra: mul!

@loops function muladd100!(_, out, a,b,c)
    let irange = eachindex(out, a,b,c)
        @inbounds @vec for i in irange
            aa, bb, cc = a[i], b[i], c[i]
            for j in 1:33
                aa = muladd(aa,bb,cc)
                bb = muladd(aa,bb,cc)
                cc = muladd(aa,bb,cc)
            end
            out[i] = muladd(aa,bb,cc)
        end
    end
end

@loops function memory_bound(_, out, a,b,c)
    let irange = eachindex(out, a)
        @inbounds @vec for i in irange
            aa, bb, cc = a[i], b[i], c[i]
            out[i] = muladd(aa,bb,cc)
        end
    end
end

function peak(fun::Fun, ops, mgrs ; F=Float32, n=5000, N=n*n) where Fun
    alloc() = zeros(F, N)
    out, a, b, c = (alloc() for i in 1:4)
    for mgr in mgrs
        (oo, aa,bb,cc) = ( (x |> mgr) for x in (out, a,b,c))
        t = minimum(1:10) do _
            @elapsed begin
                fun(mgr, oo, aa,bb,cc)
                synchronize(mgr)
            end
        end
        gflops = ops*N/t*1e-9
        @info fun mgr gflops
    end
end

versioninfo() # check JULIA_EXCLUSIVE and JULIA_NUM_THREADS

if CUDA.functional()
    include("baseline.jl")
    gpu = KernelAbstractions_GPU(CUDABackend(), CuArray)
    CUDA.versioninfo()
elseif oneAPI.functional()
    gpu = KernelAbstractions_GPU(oneAPIBackend(), oneArray)
    oneAPI.versioninfo()
else
    gpu = PlainCPU()
end

@info gpu

simd(N=8) = VectorizedCPU(N)
multi(N=8, nt=Threads.nthreads()) = MultiThread(simd(N), nt)

mgrs = (gpu, simd(), simd(16), multi(), multi(16), multi(32), multi(64))
peak(muladd100!, 200, mgrs)

mgrs = (gpu, simd(), multi())
peak(memory_bound, 4*4, mgrs ; n=5_000)

J,K,L,M = 64, 40, 42, 84
fq = randn(Float32, K, 2J,4J); # gridded data  
f = randn(Float32, 2K,2J,M); # Fourier-transformed data
fsym = randn(Float32, 2K,J,2M); # symmetric and antisymmetric parts of Fourier-transformed data
psym = randn(Float32, J,L,2M) # symmetric and antisymmetric associated Legendre polynomials
gsym = randn(Float32, 2K,L,2M); # Spectral coefficients

einsum!(g,p,f) = batched_mul!(g, f, p)

function symmetrize!(fsym, f)
    K, J, M = size(f,1), size(fsym,2), size(f, 3) 
    @assert size(fsym,1) == K
    @assert size(f,2) == 2J
    @assert size(fsym,3) == 2M
    J = size(fsym, 2)
    @inbounds for m in 1:M
        for j=1:J
            @simd for k in 1:K
                a = f[k, j, m]
                b = f[k, 2J-j+1, m]
                fsym[k,j,2m-1] = a+b
                fsym[k,j,2m] = b-a
            end
        end
    end
end
