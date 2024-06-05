using Pkg ; Pkg.activate(@__DIR__)

using InteractiveUtils
@time_imports begin
    using SIMDMathFunctions
    using BenchmarkTools
    using ThreadPinning
    using SHTnsSpheres
    using SHTnsSpheres: void
    using LoopManagers: MultiThread, VectorizedCPU, PlainCPU
    using ManagedLoops: LoopManager, @loops, @vec
    using SIMD
    using Polyester: @batch
    using LinearAlgebra
end

pinthreads(:cores)
threadinfo()

@info "================== Managed broadcast =================="

# Broadcast machinery (belongs to ManagedLoops)

Base.getindex(mgr::LoopManager, a) = Managed(mgr, a)

struct Managed{A,M,T,N} <: AbstractArray{T,N}
    mgr::M
    a::A
    Managed(mgr::M, a::A) where {M,A}= new{A, M, eltype(a), ndims(a)}(mgr, a)
end
Base.ndims(::Type{Managed{A}}) where A = ndims(A)
Base.size(ma::Managed) = size(ma.a)

function Base.copyto!(ma::Managed, bc::Broadcast.Broadcasted)
    managed_copyto!(ma.mgr, ma.a, bc)
    return ma.a
end

@loops function managed_copyto!(_, a, bc)
    let irange = eachindex(a)
        @vec for i in irange
            @inbounds a[i] = bc[i]
        end
    end
end

Base.getindex(bc::Broadcast.Broadcasted, i::SIMD.VecRange) = call(bc.f, i, bc.args...)
call(f, i, a) = @inbounds f(a[i])
call(f, i, a, b) = @inbounds f(a[i], b[i])

# benchmark

bench_bc(a,b,c) = @. a = log(exp(b)*exp(c))
bench_bc(mgr, a,b,c) = @. mgr[a] = log(exp(b)*exp(c))

a, b, c = (randn(Float32, 1000, 1000) for i=1:3)

@info "Single"
display(@benchmark bench_bc($a, $b, $c))

@info "Threads"
mgr = MultiThread()
display(@benchmark bench_bc($mgr, $a, $b, $c))

@info "SIMD"
mgr = VectorizedCPU()
display(@benchmark bench_bc($mgr, $a, $b, $c))

@info "Threads + SIMD"
mgr = MultiThread(VectorizedCPU())
display(@benchmark bench_bc($mgr, $a, $b, $c))

@info "======================= matmul ======================"

function work(repeat, a,b,c)
    for _ in 1:repeat
        mul!(a,b,c)
    end
end
work(i, repeat, a,b,c) = @views work(repeat, a[:,:,i], b[:,:,i], c[:,:,i])
work(::Nothing, repeat, a,b,c) = foreach( i->work(i,repeat,a,b,c), axes(a,3))

function twork(poly, repeat, a,b,c)
    nt = size(a,3)
    if poly
        @batch for id=1:nt
            work(id, repeat, a, b, c)
        end
    else
        Threads.@threads for id=1:nt
            work(id, repeat, a, b, c)
        end
    end
end

function bench(n, repeat)
    nt = Threads.nthreads()
    a,b,c = (randn(n,n,nt) for i=1:3)
    @info "** repeat = $repeat **"
    @info "Single"
    @btime work(nothing, $repeat, $a, $b, $c)
    @info "Threads"
    @btime(twork(false, $repeat, $a, $b, $c))
    @info "Polyester"
    @btime(twork(true, $repeat, $a, $b, $c))
end

bench(50, 2)
bench(50, 10)
bench(50, 50)
bench(50, 200)

@info "======================== Spherical harmonics =========================="

nlat = 48
@time sph = SHTnsSpheres.SHTnsSphere(nlat, 1) # single-thread
@time spht = SHTnsSpheres.SHTnsSphere(nlat)

function sphwork(spat, spec, sph)
    SHTnsSpheres.synthesis_scalar!(spat, spec, sph)
end

function bench_spectral(repeat, sph, spht)
    spec = SHTnsSpheres.shtns_alloc(Float64, Val(:scalar_spec), sph, repeat)
    spat = SHTnsSpheres.synthesis_scalar!(void, spec, sph)
    @info "** repeat = $repeat **"
    @info "Single"
    @btime sphwork($spat, $spec, $sph)
    @info "Threads"
    @btime sphwork($spat, $spec, $spht)
end

bench_spectral(30, sph, spht)
bench_spectral(100, sph, spht)
bench_spectral(300, sph, spht)

