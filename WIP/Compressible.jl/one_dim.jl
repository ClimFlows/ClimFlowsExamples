# Fully compressible solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__);
push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)
using Revise

using CFCompressible: CFCompressible, NewtonSolve, VerticalDynamics as Dyn
using CFCompressible.VerticalDynamics: VerticalEnergy, grad, total_energy, residual, tridiag_problem, bwd_Euler
using CFBatchSolvers: SingleSolvers as Solvers

includet("setup.jl");
include("config.jl");
includet("run.jl");
# includet("backward_Euler.jl")

#============================  1D model =========================#

struct OneDimModel{Gas, F}
    H::VerticalEnergy{Gas,F}
    newton::NewtonSolve
    m::Vector{F}
    S::Vector{F}
end

function CFTimeSchemes.tendencies!(::Void, scr::Void, model::OneDimModel, state::Tuple{A,A}, t, tau) where A<:Vector
    (; H, newton, m, S) = model
    Phi, W = state
    if tau>0
        Phi, W = bwd_Euler(H, newton, tau, (Phi, W, m, S))
    end
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, Phi, W, m, S)
    return (dHdW, -dHdPhi), scr
end

#============================  main program =========================#

threadinfo()
nthreads = Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = MultiThread(simd, nthreads)

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

params = map(Float64, params)
@info "Model setup..." choices params
params = (Uplanet = params.radius * params.Omega, params...)

# initial condition
loop_HPE, case = setup(choices, params, sph, mgr, HPE)
(; diags, model) = loop_HPE

H = Dyn.VerticalEnergy(model, params.gravity, params.Phis, params.pb, params.rhob)
newton = NewtonSolve(choices.newton...)

state = (Phi, W, m, S) = Dyn.initial(H, model.vcoord, case, 0.0, 0.0)
onedim = OneDimModel(H, newton, m, S)
solver = IVPSolver(TRBDF2(onedim), params.dt)

Phis=(eltype(Phi))[]
for k=1:10
    (Phi, W, _, _) = state
    @info "==================== Time step $k ======================="
    (Phi, W), t = advance!(void, solver, (Phi, W), 0., 1)
    push!(Phis, Phi[1])
end
@info Phis

# @time CFCompressible.Tests.test(H, (Phi, W, m, S))

if solver.scheme isa Midpoint
    Phi_end = copy(Phi)

    state = (Phi, W, m, S) = Dyn.initial(H, model.vcoord, case, 0.0, 0.0)
    for k=1:10
        @info "==================== Time step $k ======================="
        Phitau, Wtau = fwd_Euler(H, params.dt/2, (Phi, W, m, S))
        Phi, W = bwd_Euler(H, newton, params.dt/2, (Phitau, Wtau, m, S))
    end
    @info Phi ≈ Phi_end
end
