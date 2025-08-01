module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet

struct FCE{F, Manager, Coord, Domain, Fluid, TwoDimScalar}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    Phis::TwoDimScalar # surface geopotential
end

function FCE(model::HPE{F}, gravity::F) where F
    (; mgr, vcoord, planet, domain, gas, fcov, Phis) = model
    (; Omega, radius) = planet
    planet = ShallowTradPlanet(Omega, gravity, radius)
    return FCE(mgr, vcoord, planet, domain, gas, fcov, Phis)
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
include("tests.jl")

end # module CFCompressible
