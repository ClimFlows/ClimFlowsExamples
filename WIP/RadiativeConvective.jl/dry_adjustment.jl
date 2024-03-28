function adjust_Nlayers!(params, p, T)
    theta_mixed = potential_temperature(params, p[1], T[1])
    enthalpy = T[1] # up to factor Cp
    coef = exner(params, p[1])
    max_n = 0 # level reached by convection
    @fastmath for n in (2:params.n-1)
        theta = potential_temperature(params, p[n], T[n])
        if theta_mixed > theta
            coef = coef + exner(params, p[n])
            enthalpy = enthalpy + T[n]
            theta_mixed = enthalpy/coef
            max_n = n
        end
    end

    for m in 1:max_n
        T[m] = inverse_PT(params, p[m], theta_mixed)
    end
end
