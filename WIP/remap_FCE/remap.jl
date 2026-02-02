using CFHydrostatics.RemapHPE: vanleer, flatten, remap_density!, remap_scalar!, update_mass!
using CFDomains: mass_coordinate, data_layout
using CFTransport: remap_fluxes!
using MutatingOrNot: similar!
using ManagedLoops: @with, @vec

#========================= HPE ==========================#

function vertical_remap_HPE(model, state, scratch = void)
    layout = data_layout(model.domain)
    sph = model.domain.layer
    mass_spat = SHTnsSpheres.synthesis_scalar!(scratch.masses_spat.air, state.mass_air_spec, sph)
    massq_spat = SHTnsSpheres.synthesis_scalar!(scratch.masses_spat.consvar, state.mass_consvar_spec, sph)
    uv_spat = SHTnsSpheres.synthesis_vector!(scratch.uv_spat, state.uv_spec, sph)
    now = (
        mass = mass_spat*model.planet.radius^-2,
        massq = massq_spat,
        ux = uv_spat.ucolat,
        uy = uv_spat.ulon,
    )
    new, scratch_remapped =
        remap_HPE!(model.mgr, model.vcoord, layout, scratch.new, scratch.remapped, now)

    reshp(x) = reshape(x, size(mass_spat, 1), size(mass_spat, 2), size(mass_spat, 3))
    mass_spat .= reshp(new.mass)*model.planet.radius^2
    massq_spat .= reshp(new.massq)
    mass_air_spec = SHTnsSpheres.analysis_scalar!(state.mass_air_spec, mass_spat, sph)
    mass_consvar_spec = SHTnsSpheres.analysis_scalar!(state.mass_consvar_spec, massq_spat, sph)
    ucolat, ulon = reshp(new.ux), reshp(new.uy)
    uv_spec = SHTnsSpheres.analysis_vector!(state.uv_spec, (;ucolat, ulon), sph)

    return (; mass_air_spec, mass_consvar_spec, uv_spec) ,
        (; masses_spat=(; air=mass_spat, consvar=massq_spat), uv_spat, new, remapped=scratch_remapped)
end

function remap_HPE!(mgr, vcoord, layout, #==# new, #==# scratch, #==# now, schemes=(scalar=vanleer, momentum=vanleer))
    (; mass, massq, ux, uy) = map( x->flatten(x, layout), now)
    scheme_mq = schemes.scalar(:density, layout)
    scheme_u = schemes.momentum(:scalar, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, one(eltype(mass)))
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), scratch.flux, scratch.new_mass, #==# mass)
    # vertical transport of densities
    fluxq = similar!(scratch.fluxq, flux)
    slope = similar!(scratch.slope, massq)
    q = similar!(scratch.q, massq)
    new_massq = remap_density!(mgr, scheme_mq, new.massq, #==# fluxq, slope, q, #==# massq, mass, flux)
    # vertical transport of momentum
    new_ux = remap_scalar!(mgr, scheme_u, new.ux, #==# fluxq, slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, scheme_u, new.uy, #==# fluxq, slope, #==# uy, mass, flux)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    return (mass=new_mass, massq=new_massq, ux=new_ux, uy=new_uy), (; flux, new_mass, fluxq, slope, q)
end

#======================== FCE ======================#

function vertical_remap_FCE(model, state, tmp = void)
    layout = data_layout(model.domain)
    sph = model.domain.layer
    mass_spat = SHTnsSpheres.synthesis_scalar!(tmp.masses_spat.air, state.mass_air_spec, sph)
    massq_spat = SHTnsSpheres.synthesis_scalar!(tmp.masses_spat.consvar, state.mass_consvar_spec, sph)
    uv_spat = SHTnsSpheres.synthesis_vector!(tmp.uv_spat, state.uv_spec, sph)
    W_spat = SHTnsSpheres.synthesis_scalar!(tmp.W_spat, state.W_spec, sph)
    Phi_spat = SHTnsSpheres.synthesis_scalar!(tmp.Phi_spat, state.Phi_spec, sph)
    gradPhi_cov = SHTnsSpheres.synthesis_spheroidal!(tmp.gradPhi_cov, state.Phi_spec, sph)

    q_spat = similar!(tmp.q_spat, massq_spat)
    cov_to_horiz!(uv_spat, mass_spat, q_spat, model.mgr, model.planet.radius^-2, gradPhi_cov, W_spat, massq_spat)

    now = (
        mass = mass_spat, # per unit area
        q = q_spat,
        ux = uv_spat.ucolat,
        uy = uv_spat.ulon,
        Phi = Phi_spat,
        W = W_spat # per steradian
    )

    new, scratch_remapped =
        remap_FCE!(tmp.new, tmp.remapped, model.mgr, model.vcoord, layout, now)

    reshp(x) = reshape(x, size(mass_spat, 1), size(mass_spat, 2), size(mass_spat, 3))
    mass_spat .= reshp(new.mass)*model.planet.radius^2
    q_spat .= reshp(new.q)
    mass_air_spec = SHTnsSpheres.analysis_scalar!(state.mass_air_spec, mass_spat, sph)
    mass_consvar_spec = SHTnsSpheres.analysis_scalar!(state.mass_consvar_spec, massq_spat, sph)
    ucolat, ulon = reshp(new.ux), reshp(new.uy)
    uv_spec = SHTnsSpheres.analysis_vector!(state.uv_spec, (;ucolat, ulon), sph)

    return (; mass_air_spec, mass_consvar_spec, uv_spec) ,
        (; masses_spat=(; air=mass_spat, consvar=massq_spat), uv_spat, new, remapped=scratch_remapped)
end

function remap_FCE!(new, tmp, mgr, vcoord, layout, now, schemes=(scalar=vanleer, momentum=vanleer))
    (; mass, q, ux, uy) = map( x->flatten(x, layout), now)
    scheme_u = schemes.momentum(:scalar, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, one(eltype(mass)))
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), tmp.flux, tmp.new_mass, #==# mass)

    # vertical transport
    fluxq = similar!(tmp.fluxq, flux)
    slope = similar!(tmp.slope, q)
    new_q = remap_scalar!(mgr, scheme_u, new.q, #==# fluxq, slope, #==# q, mass, flux)
    new_ux = remap_scalar!(mgr, scheme_u, new.ux, #==# fluxq, slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, scheme_u, new.uy, #==# fluxq, slope, #==# uy, mass, flux)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    return (mass=new_mass, q=new_q, ux=new_ux, uy=new_uy), (; flux, new_mass, fluxq, slope)
end

function cov_to_horiz!(uv, mass, q, mgr, metric, gradPhi, W, massq)
    (; ucolat, ulon) = uv
    Phi_colat, Phi_lon = gradPhi
    @with mgr let (irange, jrange) = (axes(ucolat, 1), axes(ucolat, 2))
        krange = axes(ucolat, 3)
        # horizontal NH momentum (covariant)
        for (ui, Phi_i) in ((ucolat, Phi_colat), (ulon, Phi_lon))
            for j in jrange, k in krange
                @vec for i in irange
                    ui[i, j, k] = ui[i, j, k] -
                                  (Phi_i[i, j, k] * W[i, j, k] +
                                   Phi_i[i, j, k + 1] * W[i, j, k + 1]) /
                                  (2 * mass[i, j, k])
                end
            end # j,k
        end
        # compute q, apply metric factor to mass
        for j in jrange, k in krange
            @vec for i in irange
                q[i, j, k] = massq[i, j, k]/mass[i, j ,k]
                mass[i, j, k] *= metric
            end
        end
    end # @with
    return nothing
end
