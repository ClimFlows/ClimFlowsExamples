# # Spherical RSW, mimetic FD
# Shallow-water equations on a rotating sphere, mimetic finite differences on a spherical Voronoi mesh
# [Full script](VoronoiSW.jl)
# ![](VoronoiSW_3D.mp4)

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
unique!(push!(LOAD_PATH, "$(@__DIR__)/modules"))

# const debug = false
# include("preamble.jl")

@time_imports begin
    import CFDomains
    import ClimFlowsTestCases as CFTestCases
    import CFTimeSchemes
    import GFPlanets
    import CFShallowWaters
    using Filters: HyperDiffusion, filter!
    using MutatingOrNot: void
end

include("voronoi_mesh.jl")

# ## Split time scheme performing `nstep` dynamics time steps followed by filter (e.g. hyperdiffusion)
struct MyScheme{Dyn,Dissip}
    dynamics::Dyn
    dissip::Dissip
    nstep::Int
end

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

# ## Setup simulation

function setup_RSW(
    sphere;
    TestCase = CFTestCases.Williamson91{6},
    Float = Float32,
    periods = 24,
    hours_per_period = 1,
    courant = 1.5,
    Scheme = CFTimeSchemes.RungeKutta4,
    nstep = 1,
    niter_gradrot = 2,
    nu_gradrot = zero(Float),
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

    ## model setup
    planet = GFPlanets.ShallowTradPlanet(R0, Omega)
    model = CFShallowWaters.RSW(planet, sphere)
    dissip = HyperDiffusion(sphere, niter_gradrot, nu_gradrot, :vector_curl)

    ## initial condition & standard diagnostics
    state = CFShallowWaters.initialize_SW(sphere, model, CFTestCases.initial_flow, testcase)
    diags = CFShallowWaters.diagnostics(model)
    @info diags
    scratch = CFTimeSchemes.scratch_space(model, state)
    diags = CFShallowWaters.diagnostics(model)
#    return model, state, dt_dyn, MyScheme(Scheme(model), nothing, nstep), diags
    return model, diags, state, Scheme(model), dt_dyn
end

using CookBooks
(book::CookBooks.CookBook)(; kwargs...) = open(book ; kwargs...)

diagnose_pv(diags, state) = open(diags ; state) do session
    CFDomains.primal_from_dual(max.(0,session.pv), session.domain)
end

# ## Main program

## meshname, nu_gradrot = "uni.2deg.mesh.nc", 1e-15
meshname, nu_gradrot = "uni.1deg.mesh.nc", 1e-16
Float = Float32
sphere = read_mesh(meshname; Float)

model, diags, state, scheme, dt = setup_RSW(sphere; periods = 240, nu_gradrot, courant = 1.5);
fig, pv = plot_voronoi_3D(sphere, diagnose_pv(diags, state), "PV"; zoom=1)
display(fig)

let future = state
    @time for _ = 1:1000
        future = CFTimeSchemes.advance!(void, scheme, future, zero(Float), dt, void)
    end
    pv[] = diagnose_pv(diags, future)
    display(fig)
end

# @time run_movie_3D(sphere, diags, diagnose_pv, loop, "VoronoiSW_3D.mp4")
## @time run_movie_2D(sphere, diags, diagnose_pv, loop, "VoronoiSW_2D.mp4")
