using(Plots)

# Black body radiation

function planck(T, params)
    params.stephan*(T^4)
end

# Shortwave

function sw_down(params)
    flux_down_sw = zeros(params.n +1)
    tau_inf_sw = zeros(params.n +1)
    for i in 1:params.n +1
        tau_inf_sw[i] = exp(-params.c_sw*params.p_int[i]/params.mu)
        flux_down_sw[i] = tau_inf_sw[i]*params.solarc*params.mu
    end
    return flux_down_sw
end

function sw_up(params, flux_down)
    flux_up_sw = zeros(params.n +1)
    tau_0_sw = zeros(params.n +1)
    flux_up_sw[1] = params.alb*flux_down[1]
    for i in 2:params.n+1
        tau_0_sw[i] = exp(-params.c_sw*(params.Pr - params.p_int[i])/(3/5))
        flux_up_sw[i] = tau_0_sw[i]*flux_up_sw[1]
    end
    return flux_up_sw
end


# Longwave

function tau_lw(i_1, i_2, params)
    zdup = ((params.p_int[i_2])^2-(params.p_int[i_1])^2)/(2*params.g)
    return exp(-params.c_lw*sqrt(zdup))
end

function lw_down(temp, params)
    flux_down_lw = zeros(params.n +1)
    for i in 1:params.n +1  
        for k in params.n:-1:i   # k>=i 
            flux_down_lw[i] += (tau_lw(k,i, params) - tau_lw(k+1, i, params))*planck(temp[k], params)
        end
    end
    return flux_down_lw
end

function lw_up(flux_down , t_s, temp, params)
    flux_up_lw = zeros(params.n +1)
    flux_up_lw[1] = params.emissiv*planck(t_s, params) + (1-params.emissiv)*flux_down[1]
    for i in 2:params.n +1
        flux_up_lw[i] = tau_lw(i, 1, params)*flux_up_lw[1]

        for k in 1:i-1 # i>=k+1
            flux_up_lw[i] += (tau_lw(i, k+1, params) - tau_lw(i,k, params))*planck(temp[k], params)
        end
    end
    return flux_up_lw
end

function perturbation(params, temp, dt, day)

    flux_down_lw = lw_down(temp, params)

    if day <= 12

        flux_down_sw = cos(day*pi/12)sw_down(params)
        flux_up_sw = sw_up(params, flux_down_sw)

        flux_tot_sw = flux_up_sw - flux_down_sw
        
        t_s = ((flux_down_lw[1]-flux_tot_sw[1]/params.emissiv)/params.stephan)^(1/4)
        flux_up_lw = lw_up(flux_down_lw, t_s, temp, params)

        flux_tot_lw = flux_up_lw - flux_down_lw

        
    elseif 12 < day < 24
        flux_tot_sw = zeros(params.n+1)
        t_s = ((flux_down_lw[1]/params.emissiv)/params.stephan)^(1/4)
        flux_up_lw = lw_up(flux_down_lw, t_s, temp, params)

        flux_tot_lw = flux_up_lw - flux_down_lw

    end


    dT_sw = zeros(params.n)
    dT_lw = zeros(params.n)

    for i in 1:params.n
        dT_sw[i] = params.g*(flux_tot_sw[i] - flux_tot_sw[i+1])/(params.cpp*(params.p_int[i]-params.p_int[i+1]))
        dT_lw[i] = params.g*(flux_tot_lw[i] - flux_tot_lw[i+1])/(params.cpp*(params.p_int[i]-params.p_int[i+1]))
    end

    dT = dT_sw + dT_lw

    temp = temp + dT*dt

    return temp
end


potential_temperature(params, p, T) = T.*(p./params.Pr).^(-params.R/params.Cp) #computing potential temperature, p and T can be either vectors (?) or numbers
exner(params, p) = (p./params.Pr).^(params.R/params.Cp) #computing the coeficients of exner with c = (p/pr)^(R/Cp)
inverse_PT(params, p, theta) = theta*(exner(params,p)) #computing T from theta, need numbers not vectors
pressure(params) = [params.Pr - params.Pr/params.n*i for i in(0:params.n)]
temperature(params, P) = params.T0*(P./params.Pr).^(params.y*params.R/params.g)

function adjust_Nlayers!(params, p, T, theta, n)
    if theta[1] > theta[n]
        coef = exner(params, p)
        theta[1:n] .= sum(T[1:n])/sum(coef[1:n])
        T[1:n] = inverse_PT(params, p, theta[1])[1:n]
    end
end

function energy_Nlayers(params, m, T)
    energy = params.Cp*(sum(T.*m))/sum(m) # computing the energy with the formula Cp/g(T1+T2)dp
end

function simplecolumnN(params, dt, p, T, m)
    t = 0
    day = 0
    sin = []
    init_theta = potential_temperature.(Ref(params), p, T)
    thetas = []
    adj_thetas = []
    while t <= params.Tf 
        if day >= 24
            day=0
        end
        T[1] = perturbation(params,T, dt, day)[1]
        push!(thetas, potential_temperature.(Ref(params), p, T))
        theta = potential_temperature.(Ref(params), p, T)
        #before = energy_Nlayers(params, m, T) #getting the energy of the two layers before the adjustment
        for i in (2:params.n-1)
            adjust_Nlayers!(params, p, T, theta, i) #adjusting the two layers if theta 1 > theta 2
        end
        #after = energy_Nlayers(params, m, T) # energy after
        #@info "energy" before after after-before
        push!(adj_thetas, potential_temperature.(Ref(params), p, T))
        push!(sin, t)
        t += dt
        day += dt/3600
    end
    theta = potential_temperature.(Ref(params), p, T)
    plot(sin./(3600*24), [[adj_thetas[i][k] for i in (1:length(adj_thetas))] for k in (1:params.n)])
    #plot([init_theta, theta], p, label=["initial PT" "final PT"], yflip= true)
    #plot!(thetas, p, label ="warmed PT")
    #title!("Potential temperature profile")
    #xlabel!("potential temperature")
    #ylabel!("pressure (Pa)")
end

function main()
    params = (Cp = 1000.0, R = 287.0, n = 10, Pr = 101325.0, Tf = 3600*24*2, y = 0.007, T0 = 293.0, solarc = 1340/4, alb = 0.32, coefvis=0.9, coefir = 0.08, mu = 0.7,  emissiv = 0.9, cpp = 1005, g = 9.81, stephan = 5.67e-8)
    p_int = pressure(params)
    #T_int = temperature(params, p_int)
    c_sw = -.5*log(params.coefvis)/(params.Pr)
    c_lw = -log(params.coefir)/sqrt((params.Pr)^2/(2*params.g))
    params = (; params..., c_lw, c_sw,  p_int)

    p = zeros(params.n)
    m = zeros(params.n)
    T = zeros(params.n)

    for i in (1:params.n)
        p[i] = (p_int[i] + p_int[i+1])/2 #computing p in layers, only 9 values
        m[i] = (p_int[i] - p_int[i+1])/params.g #computing the mass using dp/g 
        #T[i] = (T_int[i] + T_int[i+1])/2
    end

    T = temperature(params, p)
    simplecolumnN(params, 3600, p, T, m)
end

main()
