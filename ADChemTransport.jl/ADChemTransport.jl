# # Inverse modelling
# Inversion of a chemisty-transport model similar to SpectralTransport.jl
# The inverse problem searches for the reaction rate given a time-dependent reference solution.
# This problem is solved by minimizing a loss function
# whose gradients are computed via reverse AD with Enzyme
# and fed into the Adam optimizer from Flux.Optimisers.

const start_time = time()
toc(str, t=round(time()-start_time; digits=2)) = "At t=$t s: $str"

import Pkg; Pkg.activate(@__DIR__);
@showtime import TerminalLoggers: TerminalLogger
@showtime import Flux
@showtime using Enzyme
@showtime import SHTnsSpheres
@showtime import CFTimeSchemes
@showtime using UnicodePlots
@showtime import OnlineLearningTools

using Flux.Optimise: @withprogress, @logprogress

using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, void, erase,
                        synthesis_scalar!, analysis_scalar!, analysis_vector!, divergence!,
                        sample_vector!, sample_scalar!

using CFTimeSchemes: RungeKutta4, advance!

@info toc("Local definitions")

using Logging: global_logger
global_logger(TerminalLogger())

#=============== Optimization ============#

# Adapted from `Flux.Optimise.train!`.

function train!(loss, params, grad, N, optim=Flux.Adam(0.1))
    tree = Flux.Optimisers.setup(optim, params)
    @info "initial loss" loss(params) grad(loss, params)
    @withprogress for i = 1:N
        tree, params = Flux.Optimisers.update!(tree, params, grad(loss, params))
        @logprogress i / N
    end
    @info "final loss" loss(params)
    return params
end

function enzyme_gradient(loss, params) 
    g = zero(params)
    Enzyme.autodiff(set_runtime_activity(Reverse), Const(loss), Active, Duplicated(params, g))
    return g
end

#=========== Toy chemistry-transport model ===========#

struct ToyChem{Domain, UV, S, Chem, P}
    domain::Domain
    uv::UV
    source_spec::S
    chemistry::Chem # function computing the reaction rate
    params::P # parameters for `chemistry`
end

function quadratic_reaction(params, q)
    # to make Flux happy, `params` is a length-1 array
    tau = params[1]
    return @. tau*q^2
end

function CFTimeSchemes.tendencies!(dstate, scratch, model::ToyChem, f_spec, t) 
    ToyChem_tendencies!(dstate, scratch, model, f_spec, t)
end

function ToyChem_tendencies!(dstate, scratch, model::ToyChem, f_spec, t)
    # get grid-point values of f
    f_spat = synthesis_scalar!(scratch.f_spat, f_spec, sph)
    # multiply velocity by g to obtain flux
    fluxlon   = @. scratch.fluxlon   = -f_spat*model.uv.ulon
    fluxcolat = @. scratch.fluxcolat = -f_spat*model.uv.ucolat
    # flux divergence
    flux_spat = (ucolat=fluxcolat, ulon=fluxlon)
    flux_spec = analysis_vector!(scratch.flux_spec, erase(flux_spat), sph)
    df_adv_spec = divergence!(scratch.df_adv_spec, flux_spec, sph)
    # chemistry
    df_chem_spat = model.chemistry(model.params, f_spat) # model.chemistry is non-mutating => no pre-allocation possible
    df_chem_spec = analysis_scalar!(scratch.df_chem_spec, erase(df_chem_spat), sph)
    # total
    dstate = @. dstate = df_adv_spec + model.source_spec - df_chem_spec
    scratch = (; f_spat, fluxlon, fluxcolat, flux_spec, 
                df_adv_spec, df_chem_spat, df_chem_spec)
    return dstate, scratch
end

#================ Setup inverse problem =============#

solid_body(x,y,z,lon,lat) = (sqrt(1-z*z), 0.)  # zonal solid-body "wind"

function initial_condition(f, domain)
    f_spat = sample_scalar!(void, f, domain)
    return f_spat, analysis_scalar!(void, f_spat, domain)
end

function optimal_step(dt, T)
    N = Int(ceil(T/dt))
    return N, T/N
end

function setup_online(sph, params; lmax = sph.lmax, courant = 2.0, T=1.0, velocity = solid_body, chem=quadratic_reaction)
    F(x, y, z, lon, lat) = exp(-1000 * (1 - y)^4) # source field

    Nstep, dt = optimal_step(courant / sph.lmax, T)
    @info toc("Initializing ToyChem on $sph") Nstep dt

    uv = SHTnsSpheres.sample_vector!(void, velocity, sph)
    source_spat, source_spec = initial_condition(F, sph)
    display(heatmap(source_spat))

    state = zero(source_spec)
    scheme(params) = RungeKutta4(ToyChem(sph, uv, source_spec, chem, params))
    sch = scheme(params)
    scratch = CFTistmeSchemes.scratch_space(sch, state, zero(dt))
    targets = [copy(state)] # include initial state
    for i in 1:Nstep
        CFTimeSchemes.advance!(state, sch, state, zero(dt), dt, scratch)
        push!(targets, copy(state))
    end
    display(heatmap(synthesis_scalar!(void, state, sph)))

    return scheme, OnlineLearningTools.OnlineLoss(Nstep, zero(source_spec), scratch) do i, state, scratch, params
        CFTimeSchemes.advance!(state, scheme(params), state, zero(dt), dt, scratch)
        return sum(abs2, state - targets[i+1])
    end
end

#===================== main program ======================#

@info toc("Start")

@showtime sph = SHTnsSpheres.SHTnsSphere(64);
params, guess = ([1.0], [0.0])
(scheme, loss) = setup_online(sph, params ; T=3.0);

loss(guess);
@showtime loss(guess)
enzyme_gradient(loss, guess)
@showtime enzyme_gradient(loss, guess)

optimal = train!(loss, guess, enzyme_gradient, 100)
@info toc("Done") optimal
