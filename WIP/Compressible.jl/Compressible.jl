# Fully compressible solver using spherical harmonics for horizontal discretization
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

@time_imports using CFCompressible

include("setup.jl");
include("config.jl");
include("run.jl");

#============================  main program =========================#

choices, params = experiment(choices, params)

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd

# mgr = cpu

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

rmap(fun, x) = fun(x)
rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)
params = rmap(Float64, params)

@info "Model setup..." choices params
params_testcase = (Uplanet = params.radius * params.Omega, params.testcase...)
params = (testcase=params_testcase, params...)
# initial condition

loop_HPE, case = setup(choices, params, sph, mgr, HPE)
diags_HPE, model_HPE = loop_HPE.diags, loop_HPE.model
state_HPE =  CFHydrostatics.initial_HPE(case, model_HPE)

# run_Kinnmark_Gray(params, choices, sph, mgr)
# @profview tape = simulation(merge(choices, params), loop_HPE, deepcopy(state_HPE));

#======================================================================#

newton = CFCompressible.NewtonSolve(choices.newton...)
model_FCE = CFCompressible.FCE(model_HPE, params.gravity, params.rhob, newton)
state_FCE = CFCompressible.NH_state.diagnose(model_FCE, diags_HPE, state_HPE)
diags_FCE = CFCompressible.diagnostics(model_FCE)

diags_HPE.specific_volume = (model, pressure, temperature) -> model.gas(:p, :T).specific_volume.(pressure, temperature)
diags_HPE.ulat = uv -> -uv.ucolat
diags_HPE.ulon = uv -> uv.ulon
diags_HPE.slow_fast = (model, state) -> CFTimeSchemes.tendencies!(void, void, void, model, state, 0.0, 0.0)
diags_HPE.slow = slow_fast -> slow_fast[1]
diags_HPE.fast = slow_fast -> slow_fast[2]
diags_HPE.slow_mass_air = (model, slow) -> synthesis_scalar!(void, slow.mass_air_spec, model.domain.layer)

let 
    session_HPE = open(diags_HPE ; model=model_HPE, state=state_HPE)
    session_FCE = open(diags_FCE ; model=model_FCE, state=state_FCE)

    diff(tag) = getproperty(session_HPE, tag)-getproperty(session_FCE, tag)

    function showdiff(tag) 
        var_HPE = getproperty(session_HPE, tag)[:,:,10]
        var_FCE = getproperty(session_FCE, tag)[:,:,10]
        println()
        @info "showing $tag" extrema(var_HPE) extrema(var_FCE) var_HPE ≈ var_FCE
        show(heatmap(var_FCE - var_HPE))
        show(scatterplot(var_FCE[:], var_HPE[:]))
    end

#    @info "check" session_HPE.conservative_variable ≈ session_FCE.conservative_variable
#    @info "check" session_HPE.temperature ≈ session_FCE.temperature
#    showdiff(:conservative_variable)
#    showdiff(:ulon)
#    showdiff(:specific_volume)
#    showdiff(:temperature)
#    showdiff(:Phi_dot)

    showdiff(:slow_mass_air)
end

#======================================================================#

scheme_FCE = choices.TimeScheme(model_FCE)
loop_FCE = TimeLoopInfo(sph, model_FCE, scheme_FCE, loop_HPE.remap_period, loop_HPE.dissipation, diags_FCE, choices.quicklook)

#= let 
    tau = 100.0
    slow, fast, scratch = CFTimeSchemes.tendencies!(void, void, void, model_FCE, state_FCE, 0., tau);
    slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau);
    @timev slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau);
    @profview for _ in 1:10
        slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau)
    end
end; =#

simulation(merge(choices, params), loop_FCE, state_FCE);
simulation(merge(choices, params), loop_HPE, state_HPE);

#=


include("movie.jl")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)
exit()

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#
