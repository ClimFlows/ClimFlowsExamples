# Shortwave

function sw_down(params, t)
    solarc = params.solarc * params.diurnal_cycle(t)
    flux_down_sw = zeros(params.n +1)
    tau_inf_sw = zeros(params.n +1)
    for i in 1:params.n +1
        tau_inf_sw[i] = exp(-params.c_sw*params.p_int[i]/params.mu)
        flux_down_sw[i] = tau_inf_sw[i]*solarc*params.mu
    end
    return flux_down_sw
end

function sw_up(params, flux_down)
    flux_up_sw = zeros(params.n +1)
    tau_0_sw = zeros(params.n +1)
    flux_up_sw[1] = params.alb*flux_down[1]
    for i in 2:params.n+1
        tau_0_sw[i] = exp(-params.c_sw*(params.ps - params.p_int[i])/(3/5))
        flux_up_sw[i] = tau_0_sw[i]*flux_up_sw[1]
    end
    return flux_up_sw
end

init_radiative_sw(params) = (; params..., c_sw = -.5*log(params.coefvis)/(params.ps) )
