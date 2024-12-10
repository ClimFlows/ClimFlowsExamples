# Fully compressible solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__);
push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)
using Revise

using CFCompressible
using CFCompressible: VerticalDynamics as Dyn
using CFCompressible.VerticalDynamics: VerticalEnergy, grad, total_energy, residual, tridiag_problem
using BatchSolvers: SingleSolvers as Solvers

includet("setup.jl");
include("config.jl");
includet("run.jl");
includet("backward_Euler.jl")

#============================  1D model =========================#

struct OneDimModel{Gas, F}
    H::VerticalEnergy{Gas,F}
    newton::NewtonSolve
    m::Vector{F}
    S::Vector{F}
end

function CFTimeSchemes.tendencies!(::Void, scr::Void, model::OneDimModel, state, t, tau)
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
@time CFCompressible.Tests.test(H, state)

for k=1:10
    @info "==================== Time step $k ======================="
    Phitau, Wtau = fwd_Euler(H, params.dt/2, (Phi, W, m, S))
    Phi, W = bwd_Euler(H, newton, params.dt/2, (Phitau, Wtau, m, S))
end
Phi_end = copy(Phi)

state = (Phi, W, m, S) = Dyn.initial(H, model.vcoord, case, 0.0, 0.0)
onedim = OneDimModel(H, newton, m, S)
solver = IVPSolver(choices.TimeScheme(onedim), params.dt)

Phis=(eltype(Phi))[]
for k=1:10
    @info "==================== Time step $k ======================="
    (Phi, W), t = advance!(void, solver, (Phi, W), 0., 1)
    push!(Phis, Phi[1])
end
@info Phi ≈ Phi_end


#======================#

state = CFHydrostatics.initial_HPE(case, model)
state0 = deepcopy(state)
@time tape = simulation(merge(choices, params), loop_HPE, state0);

include("movie.jl")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)
exit()

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
