struct FCE{F, Manager, Coord, Domain, Fluid, TwoDimScalar}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    Phis::TwoDimScalar # surface geopotential
end
