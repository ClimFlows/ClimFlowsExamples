# ## Setup solver

# ## Custom IVP solver splitting dynamics and filtering (e.g. hyperdiffusion)
struct MySolver{DynSolver,Dissip,F,S}
    dynsolver::DynSolver
    dissip::Dissip
    nstep::Int
    dt::F # macro time-step
    scratch::S
end

function MySolver(dyn_scheme, dissip, nstep, dt; u0 = nothing, mutating = false)
    solver = CFTimeSchemes.IVPSolver(dyn_scheme, dt; u0, mutating)
    scratch = CFDomains.scratch_space(dissip, u0.ucov)
    MySolver(solver, dissip, nstep, dt * nstep, scratch)
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

function solver(choices, params, model, state0)
    (; precision, TestCase) = choices
    ## physical parameters needed to run the model
    testcase = CFTestCases.testcase(TestCase, precision)
    @info CFTestCases.describe(testcase)
    (; R0, Omega, gH0) = testcase.params

    ## numerical parameters
    @time dx = R0 * CFDomains.laplace_dx(sphere)
    @info "Effective mesh size dx = $(round(dx/1e3)) km"
    dt_dyn = Float(courant * dx / sqrt(gH0))
    @info "Maximum dynamics time step = $(round(dt_dyn)) s"

    nstep = ceil(Int, interval / (dt_dyn * nstep_dyn))
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
    split_solver(mutating = false) =
        MySolver(dyn_scheme, dissip, nstep_dyn, dt_dyn; u0 = state0, mutating)
    dyn_solver(mutating = false) =
        CFTimeSchemes.IVPSolver(dyn_scheme, dt_dyn; u0 = state0, mutating) # unused

    return dynamics, diags, state0, split_solver, nstep, dt
end
