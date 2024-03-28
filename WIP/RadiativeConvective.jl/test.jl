include("temp.jl")

params = (n = 20, solarc = 1340, alb = 0.32, ps=101325, coefvis=0.9, coefir = 0.01, eps = 0.9, cpp = 1.0005, g = 9.81, stephan = 5.67e-8)
c_sw = -.5*log(params.coefvis)/params.ps
c_lw = -log(params.coefir)/sqrt((params.ps)^2/(2*params.g))
params = (; params..., c_lw, c_sw)

# Pressure
p_int = zeros(params.n+1)

for i in 1:params.n +1
    p_int[i]=params.ps*(1-(i/(params.n+1)))
end

params = (; params..., p_int)

# Shortwave
flux_down_sw = zeros(params.n +1)
flux_up_sw = zeros(params.n +1)
tau_0_sw = zeros(params.n +1)
tau_inf_sw = zeros(params.n +1)

for i in 1:params.n +1
    tau_inf_sw[i] = exp(-c_sw*params.p_int[i])
    flux_down_sw[i] = tau_inf_sw[i]*params.solarc
    tau_0_sw[i] = exp(-(c_sw*params.ps+c_sw*params.p_int[i]))
    flux_up_sw[i] = params.alb*tau_0_sw[i]*flux_down_sw[1]
end


println("Flux down shortwave:")
println(flux_down_sw)
println("Flux up shortwave:")
println(flux_up_sw)

println("Flux down longwave:")
test_flux_down = lw_down(zeros(params.n +1), 300*ones(params.n),params)
println(test_flux_down)
println("Flux up longwave:")
test_flux_up = lw_up(test_flux_down , 300, 300*ones(params.n), params)
println(test_flux_up)