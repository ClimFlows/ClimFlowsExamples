
struct LagrangianHPE{Manager, Domain, VCoord, Planet, Gas<:AbstractFluid, Phisurf, Fcov, F}
    manager::Manager
    domain::Domain
    # Dynamics & vertical coordinate
    vcoord::VCoord
    planet::Planet
    ptop::F
    # Thermodynamics
    gas::Gas
    # Surface geopotential
    phisurf::Phisurf
    # Coriolis factor multiplied by Jacobian
    fcov::Fcov
end

function initialize_LHPE(shell::Shell{nz, CFDomains.HVLayout}, model, fun_ps, fun_Phi, args...) where nz
    sanity_checks(model, model.vcoord)
    ulon, ulat = allocate_fields( (:scalar_spat, :scalar_spat), shell, eltype(model) )
    pressure, geopot = allocate_fields( (:scalar_spat, :scalar_spat), Domains.interfaces(shell), eltype(model) )
    domain, vcoord, ptop = shell.layer, model.vcoord, model.ptop
    # p,Phi at mass points
    let (irange, jrange) = model.backend(axes(domain.lon) ; domain, model, nz, fun_ps, fun_Phi, args, vcoord, pressure, geopot)
        for i in irange, j in jrange
            lon, lat = domain.lon[i,j], domain.lat[i,j]
            ps = fun_ps(lon, lat, args...)
            for k=0:nz
                pressure[i,j, k+1] = p = pressure_level(2k, ps, vcoord)
                geopot[i,j, k+1], _... = fun_Phi(lon, lat, p, args...)
                if k>0
                    p = pressure_level(2k-1, ps, vcoord)
                    _, ulon[i,j,k], ulat[i,j,k], _... = fun_Phi(lon, lat, p, args...)
                end
            end
            model.phisurf[i,j]=geopot[i,j,1] # surface geopotential
        end
    end
    return pressure, geopot, ulon, ulat
end

