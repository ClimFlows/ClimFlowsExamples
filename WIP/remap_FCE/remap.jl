using CFHydrostatics.RemapHPE: vanleer, flatten, remap_density!, remap_scalar!, update_mass!
using CFDomains: mass_coordinate, data_layout
using CFTransport: remap_fluxes!, mass_flux_dual!
using MutatingOrNot: similar!
using ManagedLoops: @with, @vec

#========================= Twin =========================#

function vertical_remap!(state, model::TwinModels, tmp)
    tmp_HPE = vertical_remap!(state.HPE, model.HPE, tmp.HPE)
    tmp_FCE = vertical_remap!(state.FCE, model.FCE, tmp.FCE)
    return (HPE=tmp_HPE, FCE=tmp_FCE) # == tmp
end

vertical_remap!(state, model::FCE, tmp) = CFCompressible.vertical_remap!(state, model, tmp)

#========================= HPE ==========================#

function vertical_remap!(state, model::HPE, scratch = void)
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

    return (; masses_spat=(; air=mass_spat, consvar=massq_spat), uv_spat, new, remapped=scratch_remapped)
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

#= moved to CFCompressible.RemapSpectral

function vertical_remap!(state, model::FCE, tmp)
    # steps:
    #   1 - spectral => spatial
    #   2 - covariant momentum, mass => horizontal momentum, weight
    #   3 - geopot => p_NH
    #   4 - remap
    #   5 - p_NH => Phi, ∇Phi
    #   6 - horizontal momentum, weight => covariant momentum, mass
    #   7 - spatial => spectral

    layout = data_layout(model.domain)
    @inline flat(x) = flatten(x, layout)  # (nx, ny, nz) => (nx * ny, nz)
    @inline flat(x::Union{Tuple,NamedTuple}) = map(flat, x)
    @inline flat(x...) = map(flat,x)

    sph = model.domain.layer
    #   1 - spectral => spatial
    mass_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.mass, state.mass_air_spec, sph)
    massq_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.massq, state.mass_consvar_spec, sph)
    (ux, uy) = uv_spat = SHTnsSpheres.synthesis_vector!(tmp.spat.uv, state.uv_spec, sph)
    W_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.W, state.W_spec, sph)
    Phi_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.Phi, state.Phi_spec, sph)
    gradPhi_cov = SHTnsSpheres.synthesis_spheroidal!(tmp.spat.gradPhi_cov, state.Phi_spec, sph)

    q_spat = similar!(tmp.spat.q, massq_spat)
    p_hydro = similar!(tmp.p_hydro, massq_spat)
    p_NH = similar!(tmp.p_NH, massq_spat)

    mgr, metric = model.mgr, model.planet.gravity*model.planet.radius^-2
    #   2 - covariant momentum, mass => horizontal momentum, weight
    cov_to_horiz!(uv_spat, mass_spat, q_spat, mgr, metric, gradPhi_cov, W_spat, massq_spat)
    #   3 - geopot => p_NH
    NH_pressure!(p_hydro, p_NH, model.mgr, model.gas, model.vcoord.ptop, mass_spat, q_spat, Phi_spat)

    now = (
        mass = mass_spat, # per unit area, includes gravity
        W = W_spat,       # per steradian
        q = q_spat,
        ux, uy,
        p_NH              # non-hydrostatic pressure
    )

    #   4 - remap NB : inputs (nx, ny, nz) => outputs (nx * ny, nz)
    new, scratch_remapped =
        remap_FCE!(tmp.new, tmp.remapped, model.mgr, model.vcoord, layout, now)

    #   5 - p_NH => Phi, ∇Phi
    NH_geopotential!(flat(Phi_spat, p_hydro), mgr, model.gas, model.vcoord.ptop, new.mass, new.p_NH, new.q)
    SHTnsSpheres.analysis_scalar!(state.Phi_spec, Phi_spat, sph)
    SHTnsSpheres.synthesis_spheroidal!(gradPhi_cov, state.Phi_spec, sph)

    #   6 - horizontal momentum, weight => covariant momentum, mass
    outputs = map(flat, (mass_spat, massq_spat, W_spat, ux, uy))
    horiz_to_cov!(outputs, inv(metric), new.mass, new.ux, new.uy, new.q, new.W, flat(gradPhi_cov))

    #   7 - spatial => spectral
    SHTnsSpheres.analysis_scalar!(state.mass_air_spec, mass_spat, sph)
    SHTnsSpheres.analysis_scalar!(state.mass_consvar_spec, massq_spat, sph)
    SHTnsSpheres.analysis_scalar!(state.W_spec, W_spat, sph)
    SHTnsSpheres.analysis_vector!(state.uv_spec, uv_spat, sph)

    spat = (; mass=mass_spat, massq=massq_spat, q=q_spat, uv=uv_spat, Phi=Phi_spat, W=W_spat, gradPhi_cov)
    return (; spat, p_hydro, p_NH, new, remapped=scratch_remapped) # == tmp
end

=#

#= moved to CFCompressible.remap_collocated
function cov_to_horiz!(uv, mass, q, mgr, metric, gradPhi, W, massq)
    (; ucolat, ulon) = uv
    Phi_colat, Phi_lon = gradPhi
    @with mgr let (irange, jrange) = (axes(ucolat, 1), axes(ucolat, 2))
        krange = axes(ucolat, 3)
        # horizontal NH momentum (covariant)
        for (ui, Phi_i) in ((ucolat, Phi_colat), (ulon, Phi_lon))
            for j in jrange, k in krange
                @vec for i in irange
                    ui[i, j, k] -= (Phi_i[i, j, k] * W[i, j, k] +
                                   Phi_i[i, j, k + 1] * W[i, j, k + 1]) /
                                   (2 * mass[i, j, k])
                end # i
            end # j,k
        end # colat, lon
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

function NH_pressure!(p_hydro, p_NH, mgr, gas, ptop, mass, consvar, Phi)
    pressure = gas(:v, :consvar).pressure 
    @with mgr let (irange, jrange) = (axes(p_NH, 1), axes(p_NH, 2))
        nz = size(p_NH, 3)
        for j in jrange
            # mass is per unit area, includes gravity => same unit as pressure, as in HPE
            let k=nz
                @vec for i in irange
                    vol = (Phi[i,j,k+1]-Phi[i,j,k])/mass[i,j,k]
                    p_hydro[i, j, nz] = ptop + mass[i, j, nz] / 2
                    p_NH[i,j,k] = pressure(vol, consvar[i,j,k]) - p_hydro[i,j,k]
                end
            end
            for k in nz-1:-1:1
                @vec for i in irange
                    vol = (Phi[i,j,k+1]-Phi[i,j,k])/(mass[i,j,k])                   
                    p_hydro[i, j, k] = p_hydro[i, j, k+1] + (mass[i, j, k] + mass[i, j, k+1])/2
                    p_NH[i,j,k] = pressure(vol, consvar[i,j,k]) - p_hydro[i,j,k]
                end
            end
        end
    end
    return nothing
end

function remap_FCE!(new, tmp, mgr, vcoord, layout, now, schemes=(scalar=vanleer, momentum=vanleer))
    (; mass, W, q, ux, uy, p_NH) = map( x->flatten(x, layout), now)
    momentum = schemes.momentum(:scalar, layout)
    scalar = schemes.scalar(:scalar, layout)
    density = schemes.scalar(:density, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, one(eltype(mass)))
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), tmp.flux, tmp.new_mass, #==# mass)
    # scalars
    fluxq = similar!(tmp.fluxq, flux)
    slope = similar!(tmp.slope, q)
    new_q = remap_scalar!(mgr, scalar, new.q, #==# fluxq, slope, #==# q, mass, flux)
    new_p_NH = remap_scalar!(mgr, scalar, new.p_NH, #==# fluxq, slope, #==# p_NH, mass, flux)
    new_ux = remap_scalar!(mgr, momentum, new.ux, #==# fluxq, slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, momentum, new.uy, #==# fluxq, slope, #==# uy, mass, flux)
    # densities
    mass_dual, flux_dual = mass_flux_dual!(tmp.mass_dual, tmp.flux_dual, mgr, flatten(layout), mass, flux)
#    mass_dual, flux_dual = mass_flux_dual!(mgr, flatten(layout), tmp.mass_dual, tmp.flux_dual, mass, flux)
    w = similar!(tmp.w, W)
    slopeW = similar!(tmp.slopeW, W)
    fluxW = similar!(tmp.fluxW, flux_dual)
    # new_massq = remap_density!(mgr, scheme_mq, new.massq, #==# fluxq, slope, q, #==# massq, mass, flux)
    new_W = remap_density!(mgr, density, new.W, #==# fluxW, slopeW, w, #==# W, mass_dual, flux_dual)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    # return
    tmp = (; flux, new_mass, fluxq, slope, mass_dual, flux_dual, fluxW, slopeW, w)
    return (mass=new_mass, W=new_W, q=new_q, ux=new_ux, uy=new_uy, p_NH=new_p_NH), tmp
end

# p_NH => geopot
function NH_geopotential!((Phi, p_hydro), mgr, gas, ptop, mass, p_NH, consvar)
    volume = gas(:p, :consvar).specific_volume
    nz = size(p_NH, 2)
    @with mgr let irange=axes(p_NH, 1)
        # mass is per unit area, includes gravity => same unit as pressure, as in HPE
        @vec for i in irange
            p_hydro[i, nz] = ptop + mass[i, nz] / 2
        end
        for k in nz-1:-1:1
            @vec for i in irange
                p_hydro[i, k] = p_hydro[i, k+1] + (mass[i, k] + mass[i, k+1])/2
            end
        end
        # Phi[:,1] is already set
        for k in axes(mass,2)
            @vec for i in irange
                vol = volume(p_hydro[i,k] + p_NH[i,k], consvar[i,k])
                Phi[i,k+1] = Phi[i,k] + mass[i,k]*vol
            end
        end
    end
    return nothing
end

# horizontal momentum, weight => covariant momentum, mass
function horiz_to_cov!((mass, massq, W, ux, uy), metric, new_mass, new_ux, new_uy, new_q, new_W, (Phi_x, Phi_y))
    @with mgr let (irange, krange) = axes(W)
        for i in irange, k in krange
            W[i,k] = new_W[i,k]
        end
    end
    @with mgr let (irange, krange) = axes(mass)
        for k in krange
            for i in irange 
                mass[i,k] = metric*new_mass[i,k]
                massq[i,k] = mass[i,k] * new_q[i,k]
            end 
            for (ui, Phi_i) in ((ux, Phi_x), (uy, Phi_y))
                for i in irange
                    ui[i, k] += (Phi_i[i, k] * W[i, k] +
                                Phi_i[i, k + 1] * W[i, k + 1]) /
                               (2 * mass[i, k])
                end
            end # (ux,uy)
        end # k
    end # let
end
=#

#= Moved to CFTransport 

function mass_flux_dual!(mgr, layout, mass_dual_, flux_dual_, mass, flux)
    mass_dual = similar!(mass_dual_, flux)
    flux_dual = similar!(flux_dual_, flux, size(flux,1), size(flux,2)+1)
    for i in axes(flux_dual,1)
        flux_dual[i,1] = 0
        flux_dual[i, end] = 0
        mass_dual[i,1] = mass[i,1]/2
        mass_dual[i,end] = mass[i,end]/2
    end
    for i in axes(mass_dual,1), k in crop(axes(mass,2))
        mass_dual[i,k+1] = (mass[i,k]+mass[i,k+1])/2
    end
    for i in axes(flux_dual,1), k in axes(mass,2)
        flux_dual[i,k+1] = (flux[i,k]+flux[i,k+1])/2
    end
    return mass_dual, flux_dual
end

@inline crop(ax::Base.OneTo) = Base.OneTo(ax.stop-1)

=#