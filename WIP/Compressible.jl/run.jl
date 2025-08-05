#=====  custom time loop: combines HPE solver, time integration scheme, vertical remap and hyperdiffusion ====#

# some info (essentially the time step) depends on the initial condition
struct TimeLoop{Dyn,Solver,Filter,Diags}
    model::Dyn
    solver::Solver
    remap_period::Int
    dissipation::Filter
    diags::Diags
    mutating::Bool
end

function TimeLoop(info::TimeLoopInfo, u0, time_step, mutating)
    (; model, scheme, remap_period, dissipation, diags) = info
    if mutating
        solver = IVPSolver(scheme, time_step, u0, 0.0)
    else
        solver = IVPSolver(scheme, time_step)
    end
    return TimeLoop(model, solver, remap_period, dissipation, diags, mutating)
end

# the extra parameter `dt` is for benchmarking purposes only
function run_loop(timeloop::TimeLoop, N, interval, state::State, scratch; dt = timeloop.solver.dt) where State
    (; diags, solver, model, dissipation, remap_period, mutating) = timeloop
    @assert mutating # FIXME
    t, nstep = zero(dt), round(Int, interval / dt)
    @info "Time step is $dt seconds, $nstep steps per period of $(interval/3600) hours."

    let session = open(diags ; model, state)
#        var = session.ulon[:,:,10]
        var = session.surface_pressure
        rvar = reverse(var; dims=1)
        display(heatmap(rvar))
#        show(heatmap(var-rvar))
    end

    for iter = 1:N*nstep
        state::State

        advance!(state, solver, state, t, 1)
        session = open(diags ; model, state)
#        @info "run_loop iter = $iter / $(N*nstep)" extrema(session.surface_pressure)
#        show(heatmap(session.surface_pressure))
#        show(heatmap(session.Phi_dot[:,:,10]))

#        if remap_period>0 && mod(iter, remap_period)==0
#            state = vertical_remap(model, state, scratch)
#        end
#        (; mass_consvar_spec, uv_spec) = state
#        mass_consvar_spec = dissipation.theta(mass_consvar_spec, mass_consvar_spec)
#        uv_spec = dissipation.zeta(uv_spec, uv_spec)
#        state = (; state..., mass_consvar_spec, uv_spec)
    end

    
end

function vertical_remap(model, state, scratch = void)
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
    remapped =
        CFHydrostatics.vertical_remap!(model.mgr, model, scratch.remapped, scratch, now)

    reshp(x) = reshape(x, size(mass_spat, 1), size(mass_spat, 2), size(mass_spat, 3))
    mass_spat .= reshp(remapped.mass)*model.planet.radius^2
    massq_spat .= reshp(remapped.massq)
    mass_air_spec = SHTnsSpheres.analysis_scalar!(state.mass_air_spec, mass_spat, sph)
    mass_consvar_spec = SHTnsSpheres.analysis_scalar!(state.mass_consvar_spec, massq_spat, sph)
    ucolat, ulon = reshp(remapped.ux), reshp(remapped.uy)
    uv_spec = SHTnsSpheres.analysis_vector!(state.uv_spec, (;ucolat, ulon), sph)
    return (; mass_air_spec, mass_consvar_spec, uv_spec)
end

function scratch_remap(diags, model, state)
    flatten(x) = reshape(x, size(x, 1) * size(x, 2), size(x, 3))
    flatten(x::NamedTuple) = map(flatten, x)

    uv_spat = open(diags; model, state).uv # convoluted way to allocate spatial 3D fields
    masses_spat = ( air=similar(uv_spat.ucolat), consvar=similar(uv_spat.ucolat) )

    ux = flatten(similar(uv_spat.ucolat))
    uy, mass, new_mass, massq, slope, q = (similar(ux) for _ = 1:6)
    flux = similar(mass, (size(mass, 1), size(mass, 2) + 1))
    fluxq = similar(flux)
    return (;
        masses_spat,
        uv_spat,
        flux,
        fluxq,
        new_mass,
        slope,
        q,
        remapped = (; mass, massq, ux, uy),
    )
end

function max_time_step(info::TimeLoopInfo, courant, interval, state)
    (; sphere, model, diags) = info
    # time step based on maximum sound speed and courant number `courant`, which divides `interval`
    session = open(diags; model, state)
    uv = session.uv
    cmax = maximum(session.sound_speed + @. sqrt(uv.ucolat^2 + uv.ulon^2))
    dt = model.planet.radius * courant / cmax / sqrt(sphere.lmax * sphere.lmax + 1)
    dt = divisor(dt, interval)
end

divisor(dt, T) = T / ceil(Int, T / dt)

function simulation(params, info, state0; ndays=params.ndays)
#    diag(state) = -reverse(open(diags; model, state).uv.ucolat[:, :, 1]; dims=1)
    diag(state) = reverse(open(diags; model, state).Phi_dot[:, :, 10]; dims=1)

    (; model, diags) = info
    @info "Starting simulation on $(model.mgr)."
    scratch = scratch_remap(diags, model, state0)
    (; courant, interval) = params
    dt = max_time_step(info, courant, interval, state0)
    timeloop = TimeLoop(info, state0, dt, true) # mutating, non-allocating
    N = Int(ndays * 24 * 3600 / interval)

    tape = [state0]
    state = deepcopy(state0)
        for iter = 1:N
            @info "t=$(div(interval*(iter-1),3600))h"
            @time run_loop(timeloop, 1, interval, state, scratch)
            push!(tape, deepcopy(state))
            if mod(params.interval * iter, 86400) == 0
                @info "day $(params.interval*iter/86400)"
                display(heatmap(diag(state)))
            end
        end
    return tape
end

function run_Kinnmark_Gray(params, choices, sph, mgr; ndays=1)
    params = merge(params, (; courant=4.0, ))
    choices = merge(choices, (; ndays, TimeScheme=CFTimeSchemes.KinnmarkGray{2,5}))
    loop, case = setup(choices, params, sph, mgr, HPE)
    (; diags, model) = loop
    state =  CFHydrostatics.initial_HPE(case, model)
    state0 = deepcopy(state)
    @time tape = simulation(merge(choices, params), loop_HPE, state0);
end;

