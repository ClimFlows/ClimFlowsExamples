#=====  custom time loop: combines HPE solver, time integration scheme, vertical remap and hyperdiffusion ====#

# some info (essentially the time step) depends on the initial condition
struct TimeLoop{Dyn,Solver,Filter,Diags}
    model::Dyn
    solver::Solver
    dissipation::Filter
    diags::Diags
    mutating::Bool
end

function TimeLoop(info::TimeLoopInfo, u0, time_step, mutating)
    (; model, scheme, dissipation, diags) = info
    solver = IVPSolver(scheme, time_step; u0, mutating)
    return TimeLoop(model, solver, dissipation, diags, mutating)
end

# the extra parameter `dt` is for benchmarking purposes only
function run_loop(timeloop::TimeLoop, N, interval, state, scratch; dt = timeloop.solver.dt)
    (; solver, model, dissipation, mutating) = timeloop
    @assert mutating # FIXME
    t, nstep = zero(dt), Int(interval / dt)
    for _ = 1:N*nstep
        advance!(state, solver, state, t, 1)
        mass_spec, uv_spec = vertical_remap(model, state, scratch)
        state = (;
            mass_spec = dissipation.theta(mass_spec, mass_spec),
            uv_spec = dissipation.zeta(uv_spec, uv_spec),
        )
    end
end

function vertical_remap(model, state, scratch = void)
    sph = model.domain.layer
    mass_spat = SHTnsSpheres.synthesis_scalar!(scratch.mass_spat, state.mass_spec, sph)
    uv_spat = SHTnsSpheres.synthesis_vector!(scratch.uv_spat, state.uv_spec, sph)
    now = @views (
        mass = mass_spat[:, :, :, 1],
        massq = mass_spat[:, :, :, 2],
        ux = uv_spat.ucolat,
        uy = uv_spat.ulon,
    )
    remapped =
        CFHydrostatics.vertical_remap!(model.mgr, model, scratch.remapped, scratch, now)

    reshp(x) = reshape(x, size(mass_spat, 1), size(mass_spat, 2), size(mass_spat, 3))
    mass_spat[:, :, :, 1] .= reshp(remapped.mass)
    mass_spat[:, :, :, 2] .= reshp(remapped.massq)
    mass_spec = SHTnsSpheres.analysis_scalar!(state.mass_spec, mass_spat, sph)
    uv_spec = SHTnsSpheres.analysis_vector!(
        state.uv_spec,
        (ucolat = reshp(remapped.ux), ulon = reshp(remapped.uy)),
        sph,
    )
    return (; mass_spec, uv_spec)
end

function scratch_remap(diags, model, state)
    flatten(x) = reshape(x, size(x, 1) * size(x, 2), size(x, 3))
    mass_spat = open(diags; model, state).mass
    uv_spat = open(diags; model, state).uv
    ux = flatten(similar(uv_spat.ulon))
    uy, mass, new_mass, massq, slope, q = (similar(ux) for _ = 1:6)
    flux = similar(mass, (size(mass, 1), size(mass, 2) + 1))
    fluxq = similar(flux)
    return (;
        mass_spat,
        uv_spat,
        flux,
        fluxq,
        new_mass,
        slope,
        q,
        remapped = (; mass, massq, ux, uy),
    )
end


#============== benchmark ==================#

function benchmark(choices, params, sph, mgrs)
    for mgr in mgrs
        # NB : spherical harmonics are multithread in all cases !
        @info "===== Time needed to simulate 1 day with $mgr ====="
        info, state0 = setup(choices, params, sph; mgr)
        scratch = scratch_remap(info.diags, info.model, state0)
        (; courant, interval) = params
        dt = max_time_step(info, courant, interval, state0)
        timeloop = TimeLoop(info, state0, 0.0, true) # zero time step for benchmarking
        run_benchmark(timeloop, state0, dt, interval; ndays = 0) # compile
        @time run_benchmark(timeloop, state0, dt, interval, scratch)
        #        @profview run_benchmark(timeloop, state0, dt, interval, scratch)
    end
end

function run_benchmark(timeloop, state, dt, interval, scratch = void; ndays = 1)
    N = max(1, Int(ndays * 24 * 3600 / interval))
    run_loop(timeloop, N, interval, state, scratch; dt)
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

#============== benchmark ==================#

function simulation(params, model, diags, state0; ndays=params.ndays)
    @info "Starting simulation."
    scratch = scratch_remap(diags, model, state0)
    (; courant, interval) = params
    dt = max_time_step(info, courant, interval, state0)
    timeloop = TimeLoop(info, state0, dt, true) # mutating, non-allocating
    N = Int(ndays * 24 * 3600 / interval)

    # separate thread running the simulation
    channel = Channel(spawn = true) do ch
        diag(state) = transpose(open(diags; model, state).temperature[:, :, 3])
        state = deepcopy(state0)
        for iter = 1:N
            run_loop(timeloop, 1, interval, state, scratch)
            put!(ch, diag(state))
        end
        @info "Worker: finished"
    end

    # main thread
    for i in 1:N
        @info "t=$(div(interval*i,3600))h"
        diag_t = take!(channel)
        if mod(params.interval * i, 86400) == 0
            @info "day $(i/4)"
            display(heatmap(transpose(diag_t)))
        end
    end
end
