struct TwinModels{Hydro, NH}
    HPE::Hydro
    FCE::NH
end

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere,Dyn,Scheme,Filter,Diags}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    remap_period::Int
    dissipation::Filter
    diags::Diags
    quicklook::Function # we can afford runtime dispatch when calling quicklook
end

function simulation(params, info, time_step, state0; ndays=params.ndays, interp=nothing)
    (; model, scheme, diags, quicklook) = info
    (; interval) = params
    @info "Starting simulation on $(manager(model)) for a duration of $(duration(ndays)) split into $(ndays*86400/interval) periods of $(interval/3600) hours."
    N = Int(ndays * 24 * 3600 / interval)
    tape = [state0]
    state = deepcopy(state0)
    scratch = scratch_space(scheme, state, zero(interval))
    tmp_remap = vertical_remap!(state, model, void)

    for iter = 1:N
        t = interval*(iter-1)
        @info timeinfo(div(t, 3600))
        quicklook(t, model, diags, state)
        try
            @time for j=1:div(interval, time_step)
                advance!(state, scheme, state, t+(j-1)*time_step, time_step, scratch)
                vertical_remap!(state, model, tmp_remap)
            end
        catch err
            show(err)
            @error "Caught a $(typeof(err)) !" t iter length(tape)
            return tape # return immediately with what we have been able to simulate
        end
        push!(tape, deepcopy(state))
    end
    @info timeinfo(div(interval*N, 3600))
    quicklook(interval*N, model, diags, state)
    return tape
end

manager(model) = model.mgr
manager(model::TwinModels) = manager(model.HPE)

function max_time_step(sphere, model, diags, state)
    session = open(diags; model, state)
    uv = session.uv
    cmax = maximum(session.sound_speed + @. sqrt(uv.ucolat^2 + uv.ulon^2))
    return model.planet.radius / cmax / sqrt(sphere.lmax * sphere.lmax + 1)
end

timeinfo(hours) = "t=$hours h ($(div(hours, 24)) days and $(rem(hours,24)) h)"

function duration(days)
    if days>=1
        return "$(round(days ; sigdigits=2)) days"
    else
        hours = days*24
        if hours>=1
            return "$(round(hours ; sigdigits=2)) hours"
        else
            seconds = hours*3600
            return "$(round(seconds ; sigdigits=2)) seconds"
        end
    end
end

function CFTimeSchemes.tendencies!(slow, fast, tmp, model::TwinModels, state, t, tau)
    stamp(str) = "$str (t=$t, τ=$tau)"
    slow_HPE, fast_HPE, tmp_HPE = CFTimeSchemes.tendencies!(slow.HPE, fast.HPE, tmp.HPE, model.HPE, state.HPE, t, tau)
    slow_FCE, fast_FCE, tmp_FCE = CFTimeSchemes.tendencies!(slow.FCE, fast.FCE, tmp.FCE, model.FCE, state.FCE, t, tau)
    return (HPE=slow_HPE, FCE=slow_FCE), (HPE=fast_HPE, FCE=fast_FCE), (HPE=tmp_HPE, FCE=tmp_FCE)
end

CFTimeSchemes.tendencies!(slow, fast, scratch, model::FCE, state, t, dt) = 
    CFCompressible.tendencies!(slow, fast, scratch, model, state, t, dt )

#========================= vertical remap ==========================#

function vertical_remap!(state, model::TwinModels, tmp)
    tmp_HPE = vertical_remap!(state.HPE, model.HPE, tmp.HPE)
    tmp_FCE = vertical_remap!(state.FCE, model.FCE, tmp.FCE)
    return (HPE=tmp_HPE, FCE=tmp_FCE) # == tmp
end

vertical_remap!(state, model::FCE, tmp) = CFCompressible.vertical_remap!(state, model, tmp)

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
