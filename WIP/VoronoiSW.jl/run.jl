# # Spherical RSW, mimetic FD
# Shallow-water equations on a rotating sphere, mimetic finite differences on a spherical Voronoi mesh
# [Full script](VoronoiSW.jl)
# ![](VoronoiSW_3D.mp4)

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

@time_imports begin

    using MutatingOrNot: void
    import CFDomains: CFDomains, VoronoiSphere, HyperDiffusion
    import CFTimeSchemes: CFTimeSchemes, advance!
    import CFPlanets
    import CFShallowWaters

    import ClimFlowsTestCases as CFTestCases
    using ClimFlowsData: DYNAMICO_reader
    import ClimFlowsPlots: VoronoiSphere as VSPlots
    using NetCDF, CairoMakie
end

# ## Custom IVP solver splitting dynamics and filtering (e.g. hyperdiffusion)
struct MySolver{DynSolver, Dissip, F, S}
    dynsolver::DynSolver
    dissip::Dissip
    nstep::Int
    dt::F # macro time-step
    scratch::S
end

function MySolver(dyn_scheme, dissip, nstep, dt ; u0=nothing, mutating=false)
    solver = CFTimeSchemes.IVPSolver(dyn_scheme, dt ; u0, mutating)
    scratch = CFDomains.scratch_space(dissip, u0.ucov)
    MySolver(solver, dissip, nstep, dt*nstep, scratch)
end

function filter_ucov(out, dissip, (; ghcov, ucov), dt, scratch)
    ucov_out = CFDomains.hyperdiff!(out.ucov, ucov, dissip, dissip.domain, dt, scratch, nothing)
    ghcov_out = @. out.ghcov = ghcov
    return (ghcov=ghcov_out, ucov=ucov_out)
end

"""
    future, t = advance!(future, solver, present, t, N)
    future, t = advance!(void, solver, present, t, N)
"""
function advance!(storage, solver::MySolver, state::State, t, N::Int) where State
    (; dynsolver, dissip, nstep, dt, scratch) = solver
    @assert N>0
    @assert typeof(t)==typeof(dt)
    state, t = advance!(storage, dynsolver, state, t, nstep)
    state = filter_ucov(storage, dissip, state, dt, scratch)
    for i=2:N
        state, t = advance!(storage, dynsolver, state, t, nstep)
        state = filter_ucov(storage, dissip, state, dt, scratch)
    end
    return state, t
end

# ## Setup simulation

function setup_RSW(
    sphere;
    TestCase = CFTestCases.Williamson91{6},
    Float = Float32,
    courant = 1.5,
    Scheme = CFTimeSchemes.RungeKutta4,
    nstep_dyn = 6,
    niter_gradrot = 2,
    nu_gradrot = 1e-16,
    interval = Float(3600)
)
    ## physical parameters needed to run the model
    testcase = CFTestCases.testcase(TestCase, Float)
    @info CFTestCases.describe(testcase)
    (; R0, Omega, gH0) = testcase.params

    ## numerical parameters
    @time dx = R0 * CFDomains.laplace_dx(sphere)
    @info "Effective mesh size dx = $(round(dx/1e3)) km"
    dt_dyn = Float(courant * dx / sqrt(gH0))
    @info "Maximum dynamics time step = $(round(dt_dyn)) s"

    nstep = ceil(Int, interval/(dt_dyn*nstep_dyn))
    dt = interval / nstep # macro time step, divides 'interval'
    dt_dyn = dt / nstep_dyn
    @info "Adjusted dynamics time step = $dt_dyn s"
    @info "Macro time step = $dt s"

    ## model setup
    planet = CFPlanets.ShallowTradPlanet(R0, Omega)
    dynamics = CFShallowWaters.RSW(planet, sphere)
    dissip = HyperDiffusion(sphere, niter_gradrot, nu_gradrot, :vector_curl)

    ## initial condition & standard diagnostics
    state0 = dynamics.initialize(CFTestCases.initial_flow, testcase)
    diags = dynamics.diagnostics()
    @info diags

    # split time integration
    dyn_scheme = Scheme(dynamics)
    split_solver(mutating=false) = MySolver(dyn_scheme, dissip, nstep_dyn, dt_dyn ; u0=state0, mutating)
    dyn_solver(mutating=false) = CFTimeSchemes.IVPSolver(dyn_scheme, dt_dyn ; u0=state0, mutating) # unused

    return dynamics, diags, state0, split_solver, nstep, dt
end

diagnose_pv(diags, state) = CFDomains.primal_from_dual(max.(0, open(diags; state).pv), sphere)

# ## Main program

meshname, nu_gradrot = "uni.1deg.mesh.nc", 1e-14
Float = Float32
periods, hours_per_period = 240, Float(1)
sphere = VoronoiSphere(DYNAMICO_reader(ncread, "uni.1deg.mesh.nc") ; prec=Float)
@info sphere

model, diags, state0, solver, nstep, dt = setup_RSW(sphere; nu_gradrot, courant = 1.5, interval=3600*hours_per_period);
solver! = solver(true)
@info "Macro time step = $(solver!.dt) s"
@info "Interval = $(3600*hours_per_period) s"

pv = Makie.Observable(diagnose_pv(diags, state0))

fig = VSPlots.plot_orthographic(sphere, pv);
# fig = VSPlots.plot_native_3D(sphere, pv; zoom=1);
# fig = VSPlots.plot_2D(sphere, pv; resolution=0.5);

let future = deepcopy(state0)
    record(fig, "$(@__DIR__)/PV.mp4", 1:periods) do hour
        @info "Hour $hour / $periods"
        @time CFTimeSchemes.advance!(future, solver!, future, zero(Float), nstep)
        pv[] = diagnose_pv(diags, future)
    end
end

@time let future = deepcopy(state0)
    for _ in 1:periods
        CFTimeSchemes.advance!(future, solver!, future, zero(Float), nstep)
    end
end ;
