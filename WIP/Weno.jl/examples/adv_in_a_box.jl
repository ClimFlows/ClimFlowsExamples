using LoopManagers: PlainCPU, VectorizedCPU, MultiThread, KernelAbstractions_GPU

using CFTimeSchemes: CFTimeSchemes, advance!
using MutatingOrNot: void, Void
using Weno: getrange, stencil6, stencil4, stencil2, diUn!

using Plots


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


struct Adv{F}
    msk::Matrix{UInt8}
    U::Matrix{F}
    V::Matrix{F}
end

function CFTimeSchemes.scratch_space((; msk, U, V)::Adv, mgr)
    shape = nx, ny = size(msk)
    ox = get_order(msk, 1)
    oy = get_order(msk, nx)
    flux = zeros(shape)
    return (; mgr, U, V, ox, oy, flux)
end

CFTimeSchemes.model_dstate(model::Adv{Float64}, state) =
    (trac = fill!(similar(model.U), 0),)

function CFTimeSchemes.tendencies!(dstate, model::Adv, state, scratch, t)
    (; msk) = model
    (; mgr, U, V, ox, oy, flux) = scratch
    fill!(dstate.trac, 0)
    diUn!(mgr, dstate.trac, state.trac, U, V, msk, ox, oy, flux)
    return dstate
end


function setup(shape, mgr, nh = 3)
    nx, ny = shape
    #nh = 3
    msk = zeros(UInt8, shape)
    msk[1+nh:nx-nh, 1+nh:ny-nh] .= 1

    dx = 1.0 / (nx - 2 * nh)
    dy = 1.0 / (ny - 2 * nh)

    alloc() = zeros(shape)

    state0 = (trac = alloc(),)

    for j = 1:ny, i = 1:nx
        x = (i - nh - 0.5) * dx
        y = (j - nh - 0.5) * dy
        state0.trac[i, j] =
            msk[i, j] * (mod(floor(x / (16 * dx)), 2) + mod(floor(y / (16 * dy)), 2) - 1)
    end

    U, V = alloc(), alloc()
    for j = 1:ny, i = 2:nx
        x = pi * (i - nh - 1) * dx
        y = pi * (j - nh - 0.5) * dy
        U[i, j] = -sin(x) * cos(y) * msk[i-1, j] * msk[i, j]
    end
    for j = 2:ny, i = 1:nx
        x = pi * (i - nh - 0.5) * dx
        y = pi * (j - nh - 1) * dy
        V[i, j] = cos(x) * sin(y) * msk[i, j] * msk[i, j-1]
    end


    dt = 1.0

    model = Adv(msk, U, V)

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating = false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, scheme, solver, state0
end

function main()
    shape = (262, 262)
    model, scheme, solv, state0 = setup(shape, PlainCPU())
    state = deepcopy(state0)
    solver! = solv(true)

    nite = 100
    for _ = 1:nite
        advance!(state, solver!, state, 0.0, 1)
        heatmap(state.trac) |> display
    end
end

main()
