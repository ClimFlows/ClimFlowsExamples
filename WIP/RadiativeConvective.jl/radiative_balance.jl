function radiative_balance(params, t, temp)

    flux_down_sw = sw_down(params, t)
    flux_up_sw   = sw_up(params, flux_down_sw)
    flux_down_lw = lw_down(temp, params)

    t_s = surface_temperature(params, flux_down_lw[1], flux_up_sw[1]-flux_down_sw[1])

    flux_up_lw = lw_up(flux_down_lw, t_s, temp, params)

    conv_flux(up, down, i) = (up[i]-up[i+1]) + (down[i+1]-down[i])

    @fastmath for i in 1:params.n
        Cp_dp = params.Cp*(params.p_int[i]-params.p_int[i+1])
        dT = (params.g/Cp_dp) * (conv_flux(flux_up_sw, flux_down_sw, i) + conv_flux(flux_up_lw, flux_down_lw, i)) 
        temp[i] += params.dt*dT
    end

    return t_s
end
