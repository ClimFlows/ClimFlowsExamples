struct TimeLoop{Dyn,Solver,Diags}
    model::Dyn
    solver::Solver
    remap_period::Int
    diags::Diags
    mutating::Bool
end

function TimeLoop(scheme, time_step, remap_period, diags, u0, mutating=true)
    if mutating
        solver = IVPSolver(scheme, time_step, u0, 0.0)
    else
        solver = IVPSolver(scheme, time_step)
    end
    return TimeLoop(scheme.model, solver, remap_period, diags, mutating)
end

function max_time_step(model, diags, courant, interval, state)
    # time step based on maximum sound speed and courant number `courant`, which divides `interval`
    session = open(diags; model, state)
    ke = session.kinetic_energy_i
    T, p = session.temperature_i, session.pressure_i
    sound_speed = model.gas(:p, :T).sound_speed
    cmax = maximum(@. sound_speed(p,T) + sqrt(2*ke))
    dx = model.planet.radius * CFDomains.laplace_dx(model.domain.layer)
    dt = dx * courant / cmax
    final_dt = divisor(dt, interval)
    @info "Time step:" cmax dx dt interval final_dt
    return final_dt
end

divisor(dt, T) = T / ceil(Int, T / dt)

# the extra parameter `dt` is for benchmarking purposes only
function run_loop(timeloop::TimeLoop, N, interval, state; dt = timeloop.solver.dt)
    (; solver, remap_period, mutating) = timeloop
    model = solver.scheme.model
    @assert mutating # FIXME
    if remap_period>0 
        _, scratch = CFHydrostatics.RemapVoronoi.remap!(void, void, model, state)
    else
        scratch=nothing
    end
    t, nstep = zero(dt), round(Int, interval / dt)
    @info "Time step is $dt seconds, $nstep steps per period of $(interval/3600) hours."
    for iter = 1:N*nstep
        advance!(state, solver, state, t, 1)
        if remap_period>0 && mod(iter, remap_period) ==0
            CFHydrostatics.RemapVoronoi.remap!(state, scratch, model, state)
        end
    end
end

function simulation(save, choices, params, gpu, model, diags, to_lonlat, state0; ndays=choices.ndays)
    diag(state) = open(diags; model, state, to_lonlat).surface_pressure

    @info "Initializing simulation on $(model.mgr)."
    params = rmap(choices.precision, params)
    (; courant, interval) = params
    N = Int(ndays * 24 * 3600 / interval)
    dt = max_time_step(model, diags, courant, interval, state0)

    let model = model|>gpu, state0 = state0|>gpu
        @info "Starting simulation on $(model.mgr)."
        timeloop = TimeLoop(choices.TimeScheme(model), dt, choices.remap_period, diags, state0, true) # mutating, non-allocating

        @info "Macro time step = $(timeloop.solver.dt) s"
        @info "Interval = $interval s"

        state = deepcopy(state0)
        for iter = 1:N
            @info "t=$(div(interval*(iter-1),3600))h"
            @time run_loop(timeloop, 1, interval, state)
            save(state)
            if mod(params.interval * iter, 24*3600) == 0
                @info "day $(params.interval * iter/86400)"
#                display(heatmap(diag(state)))
            end
        end
    end
    return nothing
end
