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
#    import GFDomains
    import ClimFlowsTestCases as CFTestCases
    import CFTimeSchemes
    import GFPlanets
#    import GFShallowWaters
#    import GFExperiments
#    import GFModels

#    using Filters: HyperDiffusion, filter!

#    using CairoMakie
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

CFTimeSchemes.advance!(state, scheme::MyScheme, scratch, backend) =
    advance_MyScheme!(state, scheme, scratch, backend)

function advance_MyScheme!(state, scheme, scratch, backend)
    (; dynamics, dissip, nstep) = scheme
    for _ = 1:nstep
        GFTimeSchemes.advance!(state, dynamics, scratch, backend)
    end
    ## dissipation
    ucov, _ = state
    filter!(dissip, backend, ucov, dynamics.dt * nstep)
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
    @time dx = R0 * GFDomains.laplace_dx(sphere)
    @info "Effective mesh size dx = $(round(dx/1e3)) km"
    dt_dyn = courant * dx / sqrt(gH0)
    @info "Theoretical time step = $(round(dt_dyn)) s"

    ## model setup
    planet = GFPlanets.ShallowTradPlanet(R0, Omega)
    model = GFShallowWaters.RSW(planet, sphere)
    dissip = HyperDiffusion(sphere, niter_gradrot, nu_gradrot, :vector_curl)

    ## initial condition & standard diagnostics
    state = GFModels.initialize(model, CFTestCases.initial_flow, testcase)
    scratch = GFModels.allocate_scratch(model)
    diags = GFModels.diagnostics(
        model;
        domain = sphere,
        planet = planet,
        state = state,
        scratch = scratch,
    )

    myscheme(model, dt) = MyScheme(Scheme, model, dissip, nstep, dt) # here dt is the macro step dt_dyn*nstep
    loop = GFExperiments.timeloop(
        periods,
        hours_per_period * 3600.0,
        model,
        myscheme,
        dt_dyn*nstep,
        state,
    )

    return diags, loop
end

# ## Movies

# include("voronoi.jl")

diagnose_pv(diags) = (GFDomains.primal_from_dual(diags.pv, diags.domain), "PV")

# ## Main program

## meshname, nu_gradrot = "uni.2deg.mesh.nc", 1e-15
meshname, nu_gradrot = "uni.1deg.mesh.nc", 1e-16
sphere = read_mesh(meshname; Float = Float32)

diags, loop = setup_RSW(sphere; periods = 240, nu_gradrot, courant = 2.0);
@time run_movie_3D(sphere, diags, diagnose_pv, loop, "VoronoiSW_3D.mp4")
## @time run_movie_2D(sphere, diags, diagnose_pv, loop, "VoronoiSW_2D.mp4")
