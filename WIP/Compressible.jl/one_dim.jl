using InteractiveUtils
using Revise

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

using CFCompressible: CFCompressible, NewtonSolve, VerticalDynamics as Dyn
using CFCompressible.VerticalDynamics: VerticalEnergy, grad, total_energy, residual, tridiag_problem!, bwd_Euler
using CFBatchSolvers: SingleSolvers as Solvers

includet("setup.jl");
include("config.jl");

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
        Phi, W = bwd_Euler(void, H, newton, tau, (Phi, W, m, S))
    end
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, Phi, W, m, S)
    return (dHdW, -dHdPhi), scr
end

#============================  main program =========================#

threadinfo()
nthreads = Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = MultiThread(simd, nthreads)

choices, params = experiment(choices, params)
params = rmap(Float64, params)
@info "Model setup..." choices params
params = (Uplanet = params.radius * params.Omega, params...)

model = (gas=choices.Fluid(merge(choices,params)), vcoord=SigmaCoordinate(choices.nz, params.ptop))
H = Dyn.VerticalEnergy(model, params.gravity, params.Phis, params.pb, params.rhob)
newton = NewtonSolve(merge(choices.newton, (; verbose=true))...)
case = choices.TestCase(Float64; params.testcase...)
state = (_, _, m, S) = Dyn.initial(H, model.vcoord, case, 0.0, 0.0)
onedim = OneDimModel(H, newton, m, S)

solver = IVPSolver(TRBDF2(onedim), params.dt)
Phis=(eltype(m))[]
for k=1:10
    (Phi, W, _, _) = state
    @info "==================== Time step $k ======================="
    (Phi, W), t = advance!(void, solver, (Phi, W), 0., 1)
    push!(Phis, Phi[1])
end
@info Phis
