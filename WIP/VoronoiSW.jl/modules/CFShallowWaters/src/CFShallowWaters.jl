module CFShallowWaters

using CFDomains: VoronoiSphere
using GFPlanets: ConformalPlanet
using ManagedLoops: @loops, @unroll

#=
using GFModels, GFPlanets, GFDomains, GFLoops, GFRegistries
using GFPlanets: ConformalPlanet, ShallowTradPlanet, coriolis
using GFBrackets: SW_bracket!
using GFLoops: WrappedArray, wrap_array, @unroll, @offload

import GFBrackets as BK
import GFModels as Models
import GFPlanets as Planets
=#

macro fast(code)
    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"] != "")
    return debug ? esc(code) : esc(quote
        @inbounds $code
    end)
end

abstract type AbstractSW end

#=

Models.allocate_scratch(model::AbstractSW) =
    allocate_SW_scratch(model.domain, eltype(model.domain))
Models.allocate_state(model::AbstractSW) =
    allocate_SW_state(model.domain, eltype(model.domain))
Models.backend(::AbstractSW) = GFLoops.default_backend()

allocate_SW_state(domain::SpectralDomain) = allocate_fields((:cvector, :scalar), domain)
allocate_SW_scratch(domain::SpectralDomain) =
    allocate_fields((:cvector, :scalar, :scalar), domain)

allocate_SW_state(domain::VoronoiSphere, F::Type{<:Real}) =
    allocate_fields((:vector, :scalar), domain, F)
allocate_SW_scratch(domain::VoronoiSphere, F::Type{<:Real}) =
    allocate_fields((:dual, :vector, :vector, :scalar), domain, F) # qv, qe, U, B

struct Traditional_RSW{Planet,Domain,Fcov} <: AbstractSW
    planet::Planet
    domain::Domain
    fcov::Fcov # coriolis factor multiplied by Jacobian

    function Traditional_RSW(planet::P, domain::D) where {P,D}
        fcov = coriolis_cov(eltype(domain), planet, domain)
        new{P,D,typeof(fcov)}(planet, domain, fcov)
    end
end

struct Oblate_RSW{Planet,Domain,RLambda,RPhi} <: AbstractSW
    planet::Planet
    domain::Domain
    fcov::Matrix{Float64} # coriolis factor multiplied by Jacobian
    Rlambda::RLambda      # zonal metric factor
    Rphi::RPhi            # meridional metric factor
end

RSW(planet::ShallowTradPlanet, domain) = Traditional_RSW(planet, domain)
coriolis_cov(F, planet, domain::SpectralDomain) = F.(coriolis(planet, domain.lat))
coriolis_cov(F, planet::ConformalPlanet, domain::VoronoiSphere) =
    coriolis_voronoi(F, planet, domain.lat_v, domain.Av)
coriolis_voronoi(F, planet, lat_v::W, Av::W) where {W<:WrappedArray} =
    wrap_array(coriolis_voronoi(F, planet, lat_v.data, Av.data), lat_v)
coriolis_voronoi(F, planet, lat_v, Av) = F.(coriolis(planet, lat_v) .* Av)
=#

include("initialize.jl")
include("tendencies.jl")
include("diagnostics.jl")

end # module
