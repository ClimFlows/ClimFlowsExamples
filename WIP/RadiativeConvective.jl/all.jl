# using Pkg; Pkg.activate(@__DIR__)
# using Revise

module All

using Plots

function make(T, params ; kwargs...)
 #   @info (@__LINE__) params kwargs merge(params, kwargs)
    T((getproperty(merge(params, kwargs), name) for name in fieldnames(T))...)
end

struct RadiativeConvective{F}
    # should be unnecessary
    g::F
    # thermodynamics
    Cp::F
    R::F
    Pr::F # reference pressure
    # radiative, SW
    c_sw::F
    mu::F
    alb::F
    # radiative, LW
    c_lw::F
    stephan::F
    emissiv::F
end
function RadiativeConvective(params)
    (; coefvis, ps) = params
    make(RadiativeConvective, params ;
        c_sw = -(1//2)*log(coefvis)/(ps),
        c_lw = -log(params.coefir)/sqrt((params.ps)^2/(2*params.g)))
end

struct SolarFlux{F, Fun}
    solarc::F
    cycle::Fun
end
SolarFlux(choices::NamedTuple, (; solarc)) = SolarFlux(solarc, choices.diurnal_cycle)
(sf::SolarFlux)(t) = sf.solarc * sf.cycle(t)

#======================  Shortwave ====================#

function sw_down(solarc, radconv, p_int)
    (; c_sw, mu) = radconv
    return @. solarc*mu*exp(-c_sw*p_int/mu)
end

function sw_up(radconv, flux_down, p_int)
    (; c_sw, alb) = radconv
    ps, flux_up_surf = p_int[1], alb*flux_down[1]
    return @. flux_up_surf * exp(-c_sw*(ps - p_int)*(5//3))
end

#======================  Longwave ====================#

# Black body radiation

planck(T, radconv) = radconv.stephan*(T^2)^2

function tau_lw(pi, pj, (; g, c_lw))
    zdup = (pj^2-pi^2)/(2g)
    return @fastmath exp(-c_lw*sqrt(zdup))
end

function lw_down(temp, radconv, p_int)
    F, n = eltype(temp), length(temp)
    flux_down_lw = zeros(F, n+1)
    for i in 1:n+1
        flux_down_lw_i = zero(F)
        for k in i:n   # k>=i
            pi, pk, pl = p_int[i], p_int[k], p_int[k+1]
            flux_down_lw_i += (tau_lw(pk, pi, radconv) - tau_lw(pl, pi, radconv))*planck(temp[k], radconv)
        end
        flux_down_lw[i] = flux_down_lw_i
    end
    return flux_down_lw
end

function lw_up(t_s, radconv, flux_down_lw, temp, p_int)
    F, n = eltype(temp), length(temp)
    flux_up_lw = zeros(F, n+1)
    flux_up_lw[1] = radconv.emissiv*planck(t_s, radconv) + (1-radconv.emissiv)*flux_down_lw[1]
    for i in 2:n+1
        pi = p_int[i]
        flux_up_lw_i = tau_lw(p_int[i], p_int[1], radconv)*flux_up_lw[1]
        for k in 1:i-1 # i>=k+1
            pk, pl = p_int[k], p_int[k+1]
            flux_up_lw_i += (tau_lw(pi, pl, radconv) - tau_lw(pi, pk, radconv))*planck(temp[k], radconv)
        end
        flux_up_lw[i] = flux_up_lw_i
    end
    return flux_up_lw
end

# ================= Thermodynamics ================#

potential_temperature(params, p, T) = @fastmath T*(p/params.Pr)^(-params.R/params.Cp) # computing potential temperature, p and T can be either vectors (?) or numbers

exner(params, p) = @fastmath (p./params.Pr).^(params.R/params.Cp) #computing the coeficients of exner with c = (p/pr)^(R/Cp)

inverse_PT(params, p, theta) = theta*(exner(params,p)) #computing T from theta, need numbers not vectors

pressure(params) = [params.Pr - params.Pr/params.n*i for i in(0:params.n)]

temperature(params, P) = params.T0*(P./params.Pr).^(params.y*params.R/params.g)

energy_Nlayers(params, m, T) = params.Cp*(sum(T.*m))/sum(m)

#=========== dry static adjustment ================#

function adjust_Nlayers!(radconv, p, T)
    theta_mixed = potential_temperature(radconv, p[1], T[1])
    enthalpy = T[1] # up to factor Cp
    coef = exner(radconv, p[1])
    max_n = 0 # level reached by convection so far
    for i in 2:length(T)
        theta = potential_temperature(radconv, p[i], T[i])
        if theta_mixed > theta
            coef = coef + exner(radconv, p[i])
            enthalpy = enthalpy + T[i]
            theta_mixed = enthalpy/coef
            max_n = i
        end
    end

    for m in 1:max_n
        T[m] = inverse_PT(radconv, p[m], theta_mixed)
    end
end

# ================= Radiative balance ================#

# Equilibrium surface temperature
# Ensures zero net radiative flux, given downward LW and total SW (positive upwards)
surface_temperature(params, down_lw, tot_sw) = ((down_lw-tot_sw/params.emissiv)/params.stephan)^(1//4)

function radiative_balance(solarflux, dt, radconv, temp, p_int)
    n = length(temp)
    flux_down_sw = sw_down(solarflux, radconv, p_int)
    flux_up_sw   = sw_up(radconv, flux_down_sw, p_int)
    flux_down_lw = lw_down(temp, radconv, p_int)

    t_s = surface_temperature(radconv, flux_down_lw[1], flux_up_sw[1]-flux_down_sw[1])

    flux_up_lw = lw_up(t_s, radconv, flux_down_lw, temp, p_int)

    conv_flux(up, down, i) = (up[i]-up[i+1]) + (down[i+1]-down[i])

    @fastmath for i in 1:n
        Cp_dp = radconv.Cp*(p_int[i]-p_int[i+1])
        dT = (radconv.g/Cp_dp) * (conv_flux(flux_up_sw, flux_down_sw, i) + conv_flux(flux_up_lw, flux_down_lw, i))
        temp[i] += dt*dT
    end

    return t_s
end

#===================== time loop =====================#

function temp_ev(t_f, radconv, solarflux, choices, params, temp, p_int, p_layer, c=10)
    n = length(temp)
    Y = zeros(c, n+1)        # initialisation of the vector for the plot
    i = 1                    # iterator for the plot
    l = collect(1:(t_f/c):t_f)

    for t in 1:params.dt:t_f

        t_s = radiative_balance(solarflux(t), params.dt, radconv, temp, p_int)
        choices.adjust && adjust_Nlayers!(radconv, p_layer, temp)

        if t in l
            Y[i,:] = vcat([t_s], temp)
            i += 1
        end
    end
    return temp, Y
end

#================= main() ==================#

positive(x) = (x+abs(x))/2
diurnal_cycle(t) = positive(cos(2pi*t/24/3600))
no_diurnal_cycle(t)=1

function sigma_pressure(n::Int, ps)
    p_int   = zeros(n+1)
    p_layer = zeros(n)
    for i in 1:n+1
        p_int[i]=ps*(1-(i-1)/n)
    end
    for i in 1:n
        p_layer[i] = (p_int[i]+p_int[i+1])/2
    end
    return p_int, p_layer
end

function main(ndays=1000, nn=30)
    choices = (F=Float32, n=50, adjust=true, diurnal_cycle)
    params = (dt=3600, # numerical parameters
            solarc = 1340/4,
            ps = 101325, g = 9.81,
            Pr = 101325.0, Cp = 1005, R = 287, # perfect gas
            coefvis=0.9, alb = 0.32, mu = 0.7,  # parameters for SW
            stephan = 5.67e-8, coefir = 0.08, emissiv = 0.9, # parameters for LW
            T0=293.0
            )
    params = map(choices.F, params)
    @info (@__LINE__) choices params

    radconv = RadiativeConvective(params)
    solarflux = SolarFlux(choices, params)
    @info (@__LINE__) radconv solarflux

    p_int, p_layer = sigma_pressure(choices.n, params.ps)
    temp = fill(params.T0, choices.n)
    temp, Y = temp_ev(3600*24*ndays, radconv, solarflux, choices, params, temp, p_int, p_layer)
    temp, Y = temp_ev(3600*24*2, radconv, solarflux, choices, params, temp, p_int, p_layer, 24*2)

    println(temp)
#    display(plot([Y[k,:] for k in 1:c], range(1,params.n +1)))
    display(plot(Y[:,1]))
    display(plot([potential_temperature.(Ref(radconv), p_layer[1:nn], Y[k,2:nn+1]) for k in axes(Y,1)], p_layer[1:nn] ; yflip=true))
end

end # module
using .All

# All.main()
