using InteractiveUtils
@time_imports begin
    using LoopManagers:
        SIMD, PlainCPU, VectorizedCPU, MultiThread, KernelAbstractions_GPU as GPU
    using LoopManagers.ManagedLoops: DeviceManager, synchronize

    using CFTimeSchemes: CFTimeSchemes, advance!
    using MutatingOrNot: void, Void
    using Weno: Weno, getrange, stencil6, stencil4, stencil2, diUn!

    using CairoMakie
    using GFlops

    # GPU
    using Adapt
    using CUDA: CUDA, CuArray, CUDABackend
    import KernelAbstractions
end

include("streamfunction_tools.jl")

function get_order(msk, step)
    order = similar(msk)
    fill!(order, 0)
    irange = getrange(msk, step)
    for i in irange
        s6 = sum(stencil6(msk, i, step))
        s4 = sum(stencil4(msk, i, step))
        s2 = sum(stencil2(msk, i, step))
        order[i] = (s6 == 6 ? 6 : (s4 == 4 ? 4 : (s2 == 2 ? 2 : 0)))
    end
    return order
end


struct Adv{A<:AbstractArray,M}
    msk::A
    U::A
    V::A
    mgr::M   # Loop manager
end

Adapt.adapt_structure(to::DeviceManager, adv::Adv) =
    Adv(adapt(to, adv.msk), adapt(to, adv.U), adapt(to, adv.V), to)

function CFTimeSchemes.scratch_space((; msk, U, V)::Adv, u0)
    F = eltype(u0.trac)
    shape = nx, ny = size(msk)
    ox = get_order(msk, 1)
    oy = get_order(msk, nx)
    flux = zeros(F, shape)
    return (; U, V, ox, oy, flux)
end

CFTimeSchemes.model_dstate(model::Adv, state) = (trac = fill!(similar(model.U), 0),)

function CFTimeSchemes.tendencies!(dstate, model::Adv, state, scratch, t)
    (; mgr, msk) = model
    (; U, V, ox, oy, flux) = scratch
    fill!(dstate.trac, 0)
    diUn!(mgr, dstate.trac, state.trac, U, V, msk, ox, oy, flux)
    return dstate
end

function setup(
    mgr;
    shape = (262, 262),
    nh = 3,
    courant = 1.0,
    F = Float64,
    domain = :closed,
)
    nx, ny = shape
    #nh = 3
    msk = zeros(Int8, shape)
    msk[1+nh:nx-nh, 1+nh:ny-nh] .= 1

    dx = one(F) / (nx - 2 * nh)
    dy = one(F) / (ny - 2 * nh)

    if domain == :circular
        msk = get_circular_domain(shape)
    elseif domain == :closed
        msk = get_closed_domain(shape)
    else
        @assert false "domain should be closed or circular"
    end
    U, V = get_body_rotation(msk, dx, dy)


    alloc() = zeros(F, shape)

    state0 = (trac = alloc(),)

    for j = 1:ny, i = 1:nx
        x = F(i - nh - 0.5) * dx
        y = F(j - nh - 0.5) * dy
        state0.trac[i, j] =
            F(msk[i, j]) * (mod(floor(x / (16 * dx)), 2) + mod(floor(y / (16 * dy)), 2) - 1)
    end


    dt = F(courant) / maximum(abs, U)

    model = Adv(F.(msk), U, V, mgr)

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating = false) = CFTimeSchemes.IVPSolver(scheme, dt; u0 = state0, mutating)
    return model, scheme, solver(true), state0
end

function main(solver!, state0; niter = 500)
    F = eltype(state0.trac)
    state = deepcopy(state0)
    q = Observable(state.trac)
    image(q, aspect_ratio = :equal, colormap = :viridis)
    fig = current_figure()
    display(fig)

    record(current_figure(), "box.mp4", 1:niter) do iter
        mod(iter, 50) == 0 && @info "Iteration $iter / $niter"
        advance!(state, solver!, state, zero(F), 5)
        q[] = state.trac
    end
    display(fig)
    return state
end

# should be implemented by SIMD.jl
Base.eps(::Type{SIMD.Vec{N,F}}) where {N,F} = eps(F)


# count ops ; this is possible only with PlainCPU()
@info "Counting ops..."
ops_per_point = let (F, n) = (Float64, 128)
    model, scheme, solver!, state0 = setup(PlainCPU(); shape = (n + 6, n + 6), F)
    state = deepcopy(state0)
    ops = @count_ops advance!(state, solver!, state, zero(F), 1)
    display(ops)
    GFlops.flop(ops) / length(state.trac)
end
@info "..." ops_per_point

@info "Measuring performance"
fac = 4
n = fac * 128
for mgr in (PlainCPU(), MultiThread(VectorizedCPU(16)))
    for F in (Float64, Float32)
        model, scheme, solver!, state0 = setup(mgr; shape = (n + 6, n + 6), F)
        state = deepcopy(state0)
        elapsed = minimum(1:10) do i
            (@timed advance!(state, solver!, state, zero(F), 1)).time
        end
        gflops = 1e-9 * length(state.trac) * ops_per_point / elapsed
        elapsed_per_point = "$(1e9*elapsed/length(state.trac)) ns"
        @info "$F performance" mgr gflops elapsed_per_point
    end
end

if CUDA.functional()
    fac = 16
    n = fac * 128
    for F in (Float64, Float32)
        # setup on CPU
        model, scheme, solver!, state0 = setup(PlainCPU(); shape = (n + 6, n + 6), F)
        # transfer to GPU
        gpu = GPU(CUDABackend(), CuArray)
        state0 = adapt(gpu, state0)
        solver! = adapt(gpu, solver!)

        @info typeof(state0)
        @info typeof(solver!)

        state = deepcopy(state0)
        elapsed = minimum(1:10) do i
            (@timed begin
                advance!(state, solver!, state, zero(F), 1)
                synchronize(gpu)
            end).time
        end
        gflops = 1e-9 * length(state.trac) * ops_per_point / elapsed
        elapsed_per_point = "$(1e9*elapsed/length(state.trac)) ns"
        @info "GPU performance" F gflops elapsed_per_point
    end
else
    n = fac * 128
    model, scheme, solver!, state0 =
        setup(MultiThread(VectorizedCPU(16)); shape = (n + 6, n + 6), F = Float32)
    @time state = main(solver!, state0; niter = 100)
    # @profview advance!(state, solver!, state, 0.0, 10)
end
