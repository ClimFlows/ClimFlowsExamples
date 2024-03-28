potential_temperature(params, p, T) = @fastmath T*(p/params.Pr)^(-params.R/params.Cp) # computing potential temperature, p and T can be either vectors (?) or numbers

exner(params, p) = @fastmath (p./params.Pr).^(params.R/params.Cp) #computing the coeficients of exner with c = (p/pr)^(R/Cp)

inverse_PT(params, p, theta) = theta*(exner(params,p)) #computing T from theta, need numbers not vectors

pressure(params) = [params.Pr - params.Pr/params.n*i for i in(0:params.n)]

temperature(params, P) = params.T0*(P./params.Pr).^(params.y*params.R/params.g)

function energy_Nlayers(params, m, T)
    energy = params.Cp*(sum(T.*m))/sum(m) # computing the energy with the formula Cp/g(T1+T2)dp
end

