import Pkg
const cuda_project = joinpath(@__DIR__, "CUDA", "Project.toml")
const use_cuda = Base.active_project() == cuda_project
Pkg.activate(use_cuda ? dirname(cuda_project) : joinpath(@__DIR__, "CPU"))
using InteractiveUtils

@time_imports begin
    if use_cuda
        import CUDA
        import KernelAbstractions
        CUDA.allowscalar(false)
    else
        using SIMDMathFunctions
    end
    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, KernelAbstractions_GPU, no_simd

    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, initial

    using UnicodePlots: heatmap, scatterplot
end

# Use the model loop manager when updating the spectral state.
@inline CFTimeSchemes.update!(new, model::HPE, old, args...) = CFTimeSchemes.Update.update!(new, model.mgr, old, args...)
@inline CFTimeSchemes.Update.manage(a::AbstractArray{<:Complex}, mgr::LoopManager) = no_simd(mgr)[a]

include("NCARL30.jl")

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere,Dyn,Scheme,Filter,Diags}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    remap_period::Int
    dissipation::Filter
    diags::Diags
end

#============== model setup =============#

function setup(choices, params, sph, mgr)
    case = choices.TestCase(eltype(sph.lon))
    params = merge(choices, case.params, params)
    hd_n, hd_nu = params.hyperdiff_n, params.hyperdiff_nu
    # stuff independent from initial condition
    gas = params.Fluid(params)
    # vcoord = SigmaCoordinate(params.nz, params.ptop)
    vcoord = NCARL30(params.nz, params.ptop)

    surface_geopotential(lon, lat) = initial(case, lon, lat)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    scheme = choices.TimeScheme(model)
    dissip = (
        zeta = HyperDiffusion(model.domain, hd_n, hd_nu, :vector_curl),
        theta = HyperDiffusion(model.domain, hd_n, hd_nu, :scalar),
    )
    diags = diagnostics(model)
    info = TimeLoopInfo(sph, model, scheme, choices.remap_period, dissip, diags)

    # initial condition
    state = let
        init(lon, lat) = initial(case, lon, lat)
        init(lon, lat, p) = initial(case, lon, lat, p)
        CFHydrostatics.initial_HPE(init, model)
    end

    return info, state
end

#=======================  main program =====================#

choices = (
    Fluid = IdealPerfectGas,
    TimeScheme = KinnmarkGray{2,5},
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = use_cuda ? Float32 : Float64,
    nz = 30,
    hyperdiff_n = 2,
    remap_period = 6,
    nlat = 96,
    ndays = 10,
)
params = (
    ptop = 225.52395239472398,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    hyperdiff_nu = 0, # 0.002,
    courant = 4.0,
    interval = 6 * 3600, # 6-hour intervals
)

nthreads = Threads.nthreads()

cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = use_cuda ? KernelAbstractions_GPU(CUDA.CUDABackend()) : MultiThread(simd, nthreads)

@info "Initializing spherical harmonics..."
@time sph = if use_cuda
    SHTnsSphere(choices.nlat; backend=:cuda, precision=choices.Prec, batch=choices.nz)
else
    SHTnsSphere(choices.nlat, nthreads)
end
@info sph

@info "Model setup..." choices params

params = map(choices.Prec, params)
params = (Uplanet = params.radius * params.Omega, params...)

info, state0 = setup(choices, params, sph, mgr)
(; diags, model) = info
