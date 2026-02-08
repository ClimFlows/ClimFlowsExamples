# Fully compressible solver using spherical harmonics for horizontal discretization
using Pkg; Pkg.activate(@__DIR__);
using Revise
using InteractiveUtils

includet("setup.jl");
includet("run.jl");
includet("quicklook.jl");
include("config.jl");

#============================  main program =========================#

function main(model, diags, state)
    scheme=choices.TimeScheme(model)
    loop = TimeLoopInfo(sph, model, scheme, choices.remap_period, nothing, diags, choices.quicklook)
    return simulation(merge(choices, params), loop, params.time_step, state);
end

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

tape = main(model, diags, state)
# tape = main(model.FCE, diags.FCE, state.FCE)
# tape = main(model.HPE, diags.HPE, state.HPE);

# @profview simulation(merge(choices, params, (;ndays=1/8)), loop, params.time_step, state);

serialize(joinpath(@__DIR__, "tape.jld"), (; tape, choices, params, mgr))
