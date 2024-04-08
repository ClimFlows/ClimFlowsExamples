using LoopManagers: PlainCPU, VectorizedCPU, MultiThread, KernelAbstractions_GPU

using CFTimeSchemes: CFTimeSchemes, advance!
using MutatingOrNot: void, Void
using Weno: getrange, stencil6, stencil4, stencil2, diUn!

using Plots

include("poisson.jl")


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

    dx = 1.0 / (nx - 2 * nh)
    dy = 1.0 / (ny - 2 * nh)

    alloc() = zeros(shape)

    msk = zeros(UInt8, shape)

    for j = 1:ny, i = 1:nx
        x = (i - nh - 0.5) * dx
        y = (j - nh - 0.5) * dy
        d2 = (x - 0.5)^2 + (y - 0.5)^2
        r2 = 0.5^2
        msk[i, j] = d2 < r2 ? 1 : 0

        z1 = (x - 0.5) * 2 - (y - 0.5)
        z2 = (x - 0.5) * 0.5 - (y - 0.5)
        if (z1 < 0) & (z2 > 0)
            msk[i, j] = 0
        end
    end

    msk[1:Int(round(nx / 3)), 1:Int(round(ny / 3))] .= 0

    msk[1:nh, :] .= 0
    msk[end-nh:end, :] .= 0

    msk[:, 1:nh] .= 0
    msk[:, end-nh:end] .= 0


    mskv = similar(msk)
    fill!(mskv, 0)

    for j = 2:ny, i = 2:nx
        fourcells = msk[i, j] + msk[i-1, j] + msk[i, j-1] + msk[i-1, j-1]
        mskv[i, j] = (fourcells == 4 ? 1 : 0)
    end

    A = get_A(mskv; bc = :dirichlet)

    vor = alloc()
    psi = alloc()
    for j = 1:ny, i = 1:nx
        vor[i, j] = 4 * pi * mskv[i, j] * dx * dy
    end
    mysolve!(mskv, A, vor, psi)



    state0 = (trac = alloc(),)

    for j = 1:ny, i = 1:nx
        x = (i - nh - 0.5) * dx
        y = (j - nh - 0.5) * dy
        state0.trac[i, j] =
            msk[i, j] * (mod(floor(x / (16 * dx)), 2) + mod(floor(y / (16 * dy)), 2) - 1)
    end

    U, V = alloc(), alloc()
    for j = 1:ny-1, i = 2:nx
        U[i, j] = -(psi[i, j+1] - psi[i, j]) * msk[i, j] * msk[i-1, j]
    end
    for j = 2:ny, i = 1:nx-1
        V[i, j] = (psi[i+1, j] - psi[i, j]) * msk[i, j] * msk[i, j-1]
    end


    dt = 1 / maximum(abs.(U))

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

    nite = 200
    for _ = 1:nite
        advance!(state, solver!, state, 0.0, 5)
        heatmap(state.trac, aspect_ratio = :equal) |> display
    end
end

main()
