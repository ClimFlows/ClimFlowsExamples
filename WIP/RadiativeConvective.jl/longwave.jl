# Black body radiation

function planck(T, params)
    T2=T*T
    params.stephan*T2*T2
end

# Equilibrium surface temperature
# Ensures zero net radiative flux, given downward LW and total SW (positive upwards)
surface_temperature(params, down_lw, tot_sw) = ((down_lw-tot_sw/params.emissiv)/params.stephan)^(1//4)

# Longwave

function tau_lw(i_1, i_2, (; g, p_int, c_lw))
    zdup = ((p_int[i_2])^2-(p_int[i_1])^2)/(2g)
    return @fastmath exp(-c_lw*sqrt(zdup))
end

function lw_down(temp, params)
    n = params.n
    flux_down_lw = zeros(n+1)
    for i in 1:n+1  
        flux_down_lw_i = 0.0
        for k in params.n:-1:i   # k>=i 
            flux_down_lw_i += (tau_lw(k,i, params) - tau_lw(k+1, i, params))*planck(temp[k], params)
        end
        flux_down_lw[i] = flux_down_lw_i
    end
    return flux_down_lw
end

function lw_up(flux_down , t_s, temp, params)
    flux_up_lw = zeros(params.n +1)
    flux_up_lw[1] = params.emissiv*planck(t_s, params) + (1-params.emissiv)*flux_down[1]
    for i in 2:params.n +1
        flux_up_lw_i = tau_lw(i, 1, params)*flux_up_lw[1]
        for k in 1:i-1 # i>=k+1
            flux_up_lw_i += (tau_lw(i, k+1, params) - tau_lw(i,k, params))*planck(temp[k], params)
        end
        flux_up_lw[i] = flux_up_lw_i
    end
    return flux_up_lw
end

init_radiative_lw(params) = (; params..., c_lw = -log(params.coefir)/sqrt((params.ps)^2/(2*params.g)) )
