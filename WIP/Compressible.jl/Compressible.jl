# Fully compressible solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__);
push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)
using Revise

using CFCompressible
using CFCompressible: VerticalDynamics as Dyn
using CFCompressible.VerticalDynamics: VerticalEnergy, total_energy, grad
using BatchSolvers: SingleSolvers as Solvers

includet("setup.jl");
include("config.jl");
includet("run.jl");
includet("backward_Euler.jl")

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

state = Dyn.initial(H, model.vcoord, case, 0.0, 0.0)

@time CFCompressible.Tests.test(H, state)

for k=1:10
    @info "==================== Time step $k ======================="
    tau = 100000.0
    Phitau, Wtau = fwd_Euler(H, tau, state)
    state = bwd_Euler(H, newton, tau, (Phitau, Wtau, state[3], state[4]))
end

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
