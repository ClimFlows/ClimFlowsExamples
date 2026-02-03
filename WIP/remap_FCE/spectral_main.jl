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

@showtime slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 100.);

scheme = choices.TimeScheme(model)
# solver = CFTimeSchemes.IVPSolver(scheme, 360.0, state, 0.0);
# @showtime future, t = CFTimeSchemes.advance!(void, solver, state, 0.0, 5);
# @showtime CFTimeSchemes.advance!(future, solver, state, 0.0, 5);
# @profview CFTimeSchemes.advance!(future, solver, state, 0.0, 5);

loop = TimeLoopInfo(sph, model, scheme, choices.remap_period, nothing, diags, choices.quicklook)
# tape = simulation(merge(choices, params, (; ndays=1.0, interval=4*3600)), loop, params.time_step, state);
tape = simulation(merge(choices, params, (;ndays=2)), loop, params.time_step, state);

# scheme_HPE = choices.TimeScheme(model.HPE)
# loop_HPE = TimeLoopInfo(sph, model.HPE, scheme_HPE, choices.remap_period, nothing, diags.HPE, choices.quicklook);
# tape_HPE = simulation(merge(choices, params), loop_HPE, params.time_step, state.HPE);

final_state = deepcopy(tape[end]);
remapped_HPE, tmp_HPE = vertical_remap_HPE(model.HPE, final_state.HPE, void);
remapped_FCE, tmp_FCE = vertical_remap_FCE(model.FCE, final_state.FCE, void);

# @showtime vertical_remap_HPE(model.HPE, state.HPE, scratch);
reshp(x) = reshape(x, 64, 128, size(x,2))
slice_Eq(x) = collect((x[div(size(x,1),2),:,:])')

(; gravity, radius) = model.FCE.planet
m_HPE = slice_Eq(tmp_HPE.masses_spat.air)*radius^-2 # includes gravity
m_FCE = slice_Eq(tmp_FCE.spat.mass)*radius^-2
