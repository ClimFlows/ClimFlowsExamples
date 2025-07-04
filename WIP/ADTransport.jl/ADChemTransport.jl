# # Inverse modelling
# Inversion of a chemisty-transport model similar to SpectralTransport.jl
# The inverse problem searches for the reaction rate given the final state.
# This problem is solved by minimizing a loss function
# whose gradients are computed via reverse AD with Enzyme
# and fed into the Adam optimizer from Flux.Optimisers .

# ## Preamble
const debug=false
import Pkg; Pkg.activate(@__DIR__);

const start_time = time()
toc(str, t=round(time()-start_time; digits=2)) = "At t=$t s: $str"

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

using CFTimeSchemes: RungeKutta4, IVPSolver, advance!

@info toc("Local definitions")

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

#================ Model setup =============#

solid_body(x,y,z,lon,lat) = (sqrt(1-z*z), 0.)  # zonal solid-body "wind"

function initial_condition(f, domain)
    f_spat = sample_scalar!(void, f, domain)
    return f_spat, analysis_scalar!(void, f_spat, domain)
end

function optimal_step(dt, T)
    N = Int(ceil(T/dt))
    return N, T/N
end

function setup(sph; lmax = sph.lmax, courant = 2.0, T=1.0, velocity = solid_body, chem=quadratic_reaction)
    F(x, y, z, lon, lat) = exp(-1000 * (1 - y)^4) # source field
    scheme(params) = RungeKutta4(ToyChem(sph, uv, source_spec, chem, params))

    Nstep, dt = optimal_step(courant / sph.lmax, T)
    @info toc("Initializing ToyChem on $sph") Nstep dt

    uv = SHTnsSpheres.sample_vector!(void, velocity, sph)
    source_spat, source_spec = initial_condition(F, sph)

    function forward(params) # non-mutating (Zygote)
        solver = IVPSolver(scheme(params), dt)
        spec0 = zero(eltype(params))*source_spec
        spec, t = advance!(void, solver, spec0, zero(dt), Nstep)
        return synthesis_scalar!(void, spec, sph)
    end

    function forward!(params) # mutating (Enzyme, ForwardDiff)
        spec = zero(eltype(params))*source_spec
        solver = IVPSolver(scheme(params), dt, spec, zero(dt))
        advance!(spec, solver, spec, 0.0, Nstep)
        return synthesis_scalar!(void, spec, sph)
    end
    
    function forward!!(params) # using OnlineLearningTools
        spec = zero(eltype(params))*source_spec
        scratch = CFTimeSchemes.scratch_space(scheme(params), spec, zero(dt))
        OnlineLearningTools.repeat(Nstep, spec, scratch, params) do spec, scratch, params
            CFTimeSchemes.advance!(spec, scheme(params), spec, zero(dt), dt, scratch)
            return nothing
        end
        return synthesis_scalar!(void, spec, sph)
    end

    return source_spat, forward, forward!, forward!!
end

#=============== Optimization ============#

# In ML parlance, "model" is the set of coefficients to be optimized.

struct Loss{Fun,Target}
    fun::Fun
    target::Target
end

function (loss::Loss)(model)
    (; fun, target) = loss
    predicted = fun(model)
    return sum(abs2, predicted - target)
end

function enzyme_gradient(loss, model) 
    g = zero(model)
    Enzyme.autodiff(set_runtime_activity(Reverse), Const(loss), Active, Duplicated(model, g))
    return g
end

# Adapted from `Flux.Optimise.train!`.

function train!(loss, model, grad, N, optim=Flux.Adam(0.1))
    tree = Flux.Optimisers.setup(optim, model)
    @info "initial loss" loss(model) grad(loss, model)
    @withprogress for i = 1:N
        tree, model = Flux.Optimisers.update!(tree, model, grad(loss, model))
        Flux.@logprogress i / N
    end
    @info "final loss" loss(model)
    return model
end

#===================== main program ======================#

sph = SHTnsSpheres.SHTnsSphere(64);
source0, forward, forward!, forward!! = setup(sph; T=2.0);

params = [1.0]
display(heatmap(source0))
final = forward!!(params);
display(heatmap(final))

let 
    target = forward!(params);
    @showtime forward!(params);
    loss = Loss(forward!, target)
    guess = [0.0]
    @show  loss(guess)
    @showtime enzyme_gradient(loss, guess)
    @showtime optimal = train!(loss, guess, enzyme_gradient, 100)
    @info "" optimal
end;

let 
    target = forward!!(params);
    @showtime forward!!(params);
    loss = Loss(forward!!, target)
    guess = [0.0]
    @show  loss(guess)
    @showtime enzyme_gradient(loss, guess)
    @showtime optimal = train!(loss, guess, enzyme_gradient, 100)
    @info "" optimal
end;

#================ Check gradients =================

# ## Double-check that gradients are correct
# We check the Zygote gradient by computing
# a directional gradient with ForwardDiff

Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}") #src

# complex dot product
@inline _cprod(z1, z2) = z1.re * z2.re + z1.im * z2.im
cprod(a::V, b::V) where {V<:Vector{<:Complex}} = @inline mapreduce(_cprod, +, a, b)
cprod(a::V, b::V) where {V<:Array{<:Real}} = @inline mapreduce(*, +, a, b)

function check_grad(loss, state, dstate)
    f(x) = loss(@. state + x * dstate)
    fwd_grad = ForwardDiff.derivative(f, 0.0)
    @time zyg_grad = cprod(Zygote.gradient(loss, state)[1], dstate)
    @info typeof(loss) fwd_grad zyg_grad
    return nothing
end

zygote_gradient(loss, model) = Zygote.gradient(loss, model)[1]

@showtime import ForwardDiff
@showtime import Zygote

let 
    target = forward(params);
    loss = Loss(forward, target)
    guess = [0.0]
    check_grad(loss, guess, one.(guess))
    @showtime optimal = train!(loss, guess, zygote_gradient, 100)
    @info "" optimal
end;

========================================================#
