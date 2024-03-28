using Pkg; Pkg.activate(@__DIR__)
using Revise

using Plots

include("shortwave.jl")
include("longwave.jl")
include("radiative_balance.jl")
include("thermodynamics.jl")
include("dry_adjustment.jl")

# Changes in temperature

function temp_ev(t_f, params, temp ;
                c=10) # number of curves to plot
    Y = zeros(c, params.n +1)            # initialisation of the vector for the plot
    i = 1                                # iterator for the plot
    l = [elem for elem in 1:t_f/c:t_f]

    for t in 1:params.dt:t_f

        t_s = radiative_balance(params, t, temp)
        params.adjust && adjust_Nlayers!(params, params.p_layer, temp)

        if t in l
            Y[i,:] = vcat([t_s], temp)
            i += 1
        end
    end
    return temp, Y
end

# Execution

positive(x) = (x+abs(x))/2
diurnal_cycle(t) = positive(cos(t/24/3600*2pi))
no_diurnal_cycle(t)=1

function main()
    params = (n = 50, dt=3600, # numerical parameters
            solarc = 1340/4, diurnal_cycle = diurnal_cycle,
            ps = 101325, g = 9.81,
            Pr = 101325.0, Cp = 1005, R = 287, # perfect gas
            coefvis=0.9, alb = 0.32, mu = 0.7,  # parameters for SW
            stephan = 5.67e-8, coefir = 0.08, emissiv = 0.9, # parameters for LW
            adjust=true,
            )
    params = init_radiative_lw(params)
    params = init_radiative_sw(params)

    # Pressure
    p_int   = zeros(params.n+1)
    p_layer = zeros(params.n)

    for i in 1:params.n +1
        p_int[i]=params.ps*(1-(i/(params.n+1)))
    end
    for i in 1:params.n
        p_layer[i] = (p_int[i]+p_int[i+1])/2
    end

    params = (; params..., p_int, p_layer)

    temp = 293*ones(params.n)
    @profview temp, Y = temp_ev(3600*24*1000, params, temp)
    temp, Y = temp_ev(3600*24*2, params, temp, c=24*2)

    println(temp)
#    display(plot([Y[k,:] for k in 1:c], range(1,params.n +1)))
    nn=30
    display(plot(Y[:,1]))

    display(plot([potential_temperature.(Ref(params), params.p_layer[1:nn], Y[k,2:nn+1]) for k in axes(Y,1)], params.p_layer[1:nn] ; yflip=true))

end

main()
