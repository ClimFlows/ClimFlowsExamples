module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet

struct SoftSurface{T, A<:AbstractMatrix{T}} 
end

struct FCE{F, Manager, Coord, Domain, Fluid, TwoDimScalar<:AbstractMatrix{F}}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    # bottom boundary condition: p = pb - rhob*(Phi-Phis)
    Phis::TwoDimScalar # target surface geopotential
    pb::TwoDimScalar # surface pressure when Phi == Phis
    rhob::F # larger rhob makes smaller Phi-Phis but stiffer system
end

function FCE(model::HPE{F}, gravity::F, pb, rhob) where F
    (; mgr, vcoord, planet, domain, gas, fcov, Phis) = model
    (; radius, Omega) = planet
    planet = ShallowTradPlanet(radius, Omega, gravity)
    return FCE(mgr, vcoord, planet, domain, gas, fcov, Phis, pb, rhob)
end

# implemented in Dynamics
"""
slow, fast, scratch = FCE_tendencies!(slow, fast, scratch, model, layer, state, tau)
"""
function FCE_tendencies!
end

"""
slow, fast, scratch = tendencies!(slow, fast, scratch, model, state, t, tau)
"""
tendencies!(slow, fast, scratch, model::FCE, state, _, tau) = FCE_tendencies!(slow, fast, scratch, model, model.domain.layer, state, tau)

include("vertical_dynamics.jl")
include("horizontal_energies.jl")
include("dynamics.jl")
include("NH_state.jl")
include("diagnostics.jl")
include("tests.jl")

diagnostics(::FCE) = Diagnostics.diagnostics()

end # module CFCompressible
