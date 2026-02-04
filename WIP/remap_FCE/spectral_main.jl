# Fully compressible solver using spherical harmonics for horizontal discretization
using Pkg; Pkg.activate(@__DIR__);
using Revise
using InteractiveUtils

includet("setup.jl");
includet("run.jl");
includet("quicklook.jl")
includet("remap.jl")
include("config.jl");

# rmap(fun, x) = fun(x)
# rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)

# synth(x) = synth(x, eltype(x))
# synth(x, ::Type) = x
# synth(x, ::Type{<:Complex}) = synthesis_scalar!(void, x, sph)

# to_deg(rad) = (180/pi)*rad

# Ldiff(x,y) = round(Linf(x-y)/max(Linf(x),Linf(y)); sigdigits=2)

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd
# mgr = cpu

choices, params = experiment(choices, params)
params = rmap(Float64, params)
params_testcase = (Uplanet = params.radius * params.Omega, params.testcase...)
params = (testcase=params_testcase, params...)    

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
@showtime sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

model, state, diags = setup(choices, params, sph, mgr)
scheme = choices.TimeScheme(model)
loop = TimeLoopInfo(sph, model, scheme, choices.remap_period, nothing, diags, choices.quicklook)
# @profview simulation(merge(choices, params, (;ndays=1/8)), loop, params.time_step, state);
tape = simulation(merge(choices, params), loop, params.time_step, state);
