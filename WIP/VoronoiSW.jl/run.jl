# # Spherical RSW, mimetic FD
# Shallow-water equations on a rotating sphere, mimetic finite differences on a spherical Voronoi mesh
# [Full script](VoronoiSW.jl)
# ![](VoronoiSW_3D.mp4)

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
unique!(push!(LOAD_PATH, "$(@__DIR__)/modules"))
using InteractiveUtils

# const debug = false
# include("preamble.jl")

@time_imports begin
    import CFDomains: CFDomains, VoronoiSphere
    import ClimFlowsTestCases as CFTestCases
    import CFTimeSchemes
    import GFPlanets
    import CFShallowWaters
    using MutatingOrNot: void
    using ClimFlowsData: DYNAMICO_reader
    import ClimFlowsPlots: VoronoiSphere as VSPlots
    using CookBooks

    using Filters: HyperDiffusion, filter!
    using NetCDF, CairoMakie
end

# belongs to CookBooks
(book::CookBooks.CookBook)(; kwargs...) = open(book ; kwargs...)

# ## Split time scheme performing `nstep` dynamics time steps followed by filter (e.g. hyperdiffusion)
struct MyScheme{Dyn,Dissip}
    dynamics::Dyn
    dissip::Dissip
    nstep::Int
end

#=
MyScheme(Scheme, model, dissip, nstep, dt) =
    MyScheme(Scheme(model, dt / nstep), dissip, nstep)

Base.show(io::IO, scheme::MyScheme{Dyn, Dissip}) where {Dyn, Dissip} = print(io,
    "MyScheme\n",
    "  dynamics::$Dyn\n",
    "  dissip::$Dissip\n",
    "  nstep = $(scheme.nstep)\n")

CFTimeSchemes.advance!(future, scheme::MyScheme, state, t, dt, scratch) =
    advance_MyScheme!(future, scheme, state, t, dt, scratch)

function advance_MyScheme!(future, scheme, state, t, dt, scratch)
    (; dynamics, dissip, nstep) = scheme
#    for _ = 1:nstep
        future = CFTimeSchemes.advance!(future, dynamics, state, t, dt, scratch)
#    end
    ## dissipation
#    (; ucov) = state
#    filter!(dissip, backend, ucov, dynamics.dt * nstep)
end
=#

# ## Setup simulation

function setup_RSW(
    sphere;
    TestCase = CFTestCases.Williamson91{6},
    Float = Float32,
    courant = 1.5,
    Scheme = CFTimeSchemes.RungeKutta4,
    nstep = 1,
    niter_gradrot = 2,
    nu_gradrot = zero(Float),
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
    @info "Theoretical time step = $(round(dt_dyn)) s"
    nstep = ceil(Int, interval/dt_dyn)
    dt = interval / nstep
    @info "Adjusted time step = $dt"

    ## model setup
    planet = GFPlanets.ShallowTradPlanet(R0, Omega)
    model = CFShallowWaters.RSW(planet, sphere)
    dissip = HyperDiffusion(sphere, niter_gradrot, nu_gradrot, :vector_curl)

    ## initial condition & standard diagnostics
    state0 = model.initialize(CFTestCases.initial_flow, testcase)
    diags = model.diagnostics()
    @info diags

    scheme = Scheme(model)
    solver(mutating=false) = CFTimeSchemes.IVPSolver(scheme, dt_dyn ; u0=state0, mutating)
    return model, diags, state0, scheme, solver, nstep, dt_dyn
#    return model, state, dt_dyn, MyScheme(Scheme(model), nothing, nstep), diags
end

diagnose_pv(diags, state) = CFDomains.primal_from_dual(max.(0, diags(; state).pv), sphere)

# ## Main program

## meshname, nu_gradrot = "uni.2deg.mesh.nc", 1e-15
meshname, nu_gradrot = "uni.1deg.mesh.nc", 1e-16
Float = Float32
periods, hours_per_period = 240, Float(1)
sphere = VoronoiSphere(DYNAMICO_reader(ncread, "uni.1deg.mesh.nc") ; prec=Float)
@info sphere

model, diags, state0, scheme, solver, nstep, dt = setup_RSW(sphere; nu_gradrot, courant = 1.5);
solver! = solver(true)

pv = Makie.Observable(diagnose_pv(diags, state0))
fig = VSPlots.plot_orthographic(sphere, pv);
# fig = VSPlots.plot_native_3D(sphere, pv; zoom=1);
# fig = VSPlots.plot_2D(sphere, pv; resolution=0.25);

let future = deepcopy(state0)
    record(fig, "$(@__DIR__)/PV.mp4", 1:periods) do hour
        @info "Hour $hour / $periods"
        @time CFTimeSchemes.advance!(future, solver!, future, zero(Float), nstep)
        pv[] = diagnose_pv(diags, future)
    end
end
