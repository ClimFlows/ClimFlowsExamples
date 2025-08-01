module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres:
                    analysis_scalar!,
                    synthesis_scalar!,
                    analysis_vector!,
                    synthesis_vector!,
                    synthesis_spheroidal!,
                    divergence!,
                    curl!
using ManagedLoops: @with, @vec

function diagnostics()
    return CookBook(;
                    # independent from vertical coordinate
                    uv,
                    pressure,
                    temperature,
                    sound_speed,
                    # depend on vertical coordinate
                    masses,
                    conservative_variable,
    #=
                    surface_pressure,
                    geopotential,
                    Omega,
                    Phi_dot,
                    # dX is the local time derivative of X, assuming a Lagrangian vertical coordinate
                    dmasses,
                    duv,
                    dulon,
                    dulat,
                    dpressure,
                    dgeopotential,
                    # intermediate computations
                    dstate,
                    ps_spec,
                    gradPhi_cov,
                    # mostly for debugging
                    dstate_all,
                    gradmass,
                    ugradps,
                    gradPhi,
    =#
                    )
end

#=================== independent from vertical coordinate ==============#

# same as HPE

function sound_speed(model, pressure, temperature)
    return model.gas(:p, :T).sound_speed.(pressure, temperature)
end

function temperature(model, pressure, conservative_variable)
    return model.gas(:p, :consvar).temperature.(pressure, conservative_variable)
end

conservative_variable(masses) = @. masses.consvar / masses.air

# FCE-specicific

function uv(model, state) # FIXME: HPE version
    (; ucolat, ulon) = synthesis_vector!(void, map(copy, state.uv_spec), model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat=invrad * ucolat, ulon=invrad * ulon)
end

function pressure(model, masses) # FIXME: HPE version
    mass = masses.air
    p = similar(mass)
    ptop = model.vcoord.ptop # avoids capturing `model`
    @with model.mgr let (irange, jrange) = (axes(p, 1), axes(p, 2))
        nz = size(p, 3)
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + mass[i, j, nz] / 2
                for k in nz:-1:2
                    p[i, j, k - 1] = p[i, j, k] + (mass[i, j, k] + mass[i, j, k - 1]) / 2
                end
            end
        end
    end
    return p
end

#======================= depend on vertical coordinate ======================#

# same as HPE

function masses(model, state)
    fac, sph = model.planet.radius^-2, model.domain.layer
    return (air=synthesis_scalar!(void, fac * state.mass_air_spec, sph),
            consvar=synthesis_scalar!(void, fac * state.mass_consvar_spec, sph))
end

end # module
