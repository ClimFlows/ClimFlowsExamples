struct TwinModels{Hydro, NH}
    HPE::Hydro
    FCE::NH
end

function CFTimeSchemes.tendencies!(slow, fast, tmp, model::TwinModels, state, t, tau)
    stamp(str) = "$str (t=$t, τ=$tau)"
    slow_HPE, fast_HPE, tmp_HPE = CFTimeSchemes.tendencies!(slow.HPE, fast.HPE, tmp.HPE, model.HPE, state.HPE, t, tau)
    slow_FCE, fast_FCE, tmp_FCE = CFTimeSchemes.tendencies!(slow.FCE, fast.FCE, tmp.FCE, model.FCE, state.FCE, t, tau)
    return (HPE=slow_HPE, FCE=slow_FCE), (HPE=fast_HPE, FCE=fast_FCE), (HPE=tmp_HPE, FCE=tmp_FCE)
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
                # run_loop(timeloop, 1, interval, state, scratch)
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
