function fwd_Euler(H, tau, state)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, state...)
    Phitau = @. Phi + tau * dHdW
    Wtau = @. W - tau * dHdPhi
    return Phitau, Wtau
end

square(x) = x * x
L2(x) = sqrt(sum(square, x))
mindiff(x) = minimum(k->x[k+1]-x[k], 1:length(x)-1)
check(x) = if isnan(L2(x)) || isinf(L2(x)) 
    @info x 
    error()
end

function bwd_Euler(H, tau, state)
    (; gravity) = H
    (Phi_star, W_star, m, S) = state
    DPhi, Phi, W = zero(Phi_star), copy(Phi_star), copy(W_star)
    @. Phi += H.Phis - Phi[1] # lift initial guess to satisfy PHi[1]=Phis
    for iter in 1:5
        check(Phi)
        check(W)
        check(m)
        check(S)
        check(Phi_star)
        check(W_star)
        rPhi, rW = residual(H, tau, (Phi, W, m, S), Phi_star, W_star)
        check(rPhi)
        check(rW)
        A, B, R = tridiag_problem(H, tau, Phi, W, m, S, rPhi, rW)
        check(A)
        check(B)
        check(R)
#        @info "residuals" rPhi rW R
#        TD = SymTridiagonal(B, -A)
#        eigenvalues = eigvals(TD) 
#        @info extrema(eigenvalues)
#        C = cholesky(TD)
        dPhi = Thomas_flip(A, B, R, true)
        RR = check_Thomas(A, B, R, dPhi)
        # dPhi = C \ R
        @. DPhi += dPhi
        @info "Residuals" iter L2(R) L2(RR) L2(rPhi) L2(rW)

        @. Phi = Phi_star + DPhi

        # W = ml * (Phi-Phi_star) / (tau*g^2)
        Nz = length(m)
        let l = 1, ml = m[l] / 2
            W[l] = (ml * DPhi[l]) / (tau * gravity^2)
        end
        for l in 2:Nz
            ml = (m[l - 1] + m[l]) / 2
            W[l] = (ml * DPhi[l]) / (tau * gravity^2)
        end
        let l = Nz + 1, ml = m[l - 1] / 2
            W[l] = (ml * DPhi[l]) / (tau * gravity^2)
        end
    end
    return Phi, W, m, S
end

function residual(H, tau, state, Phi_star, W_star)
    # we are solving 
    #    x = x⋆ + τf(x)
    # state is the current guess
    # the residual is τ f(x) + (x⋆-x)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, state...)
    dPhi = @. (Phi_star - Phi) + tau * dHdW
    dW = @. (W_star - W) - tau * dHdPhi
    return dPhi, dW
end

function solve(H, tau, (Phi, W, m, S), rPhi, rW)
    return dPhi
end

function tridiag_problem(H, tau, Phi, W, m, S, rPhi, rW)
    (; gas, gravity, rhob, J) = H
    Nz = length(m)
    ml, A, B, R = map(x->fill(NaN, size(x)), (Phi, m, Phi, Phi))

    # ml
    ml[1] = m[1] / 2
    for l in 2:Nz
        ml[l] = (m[l - 1] + m[l]) / 2
    end
    ml[Nz + 1] = m[Nz] / 2

    # off-diagonal coeffcient A[k]
    for k in 1:Nz
        mk = m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / mk # specific volume
        consvar = S[k] / m[k] # conservative variable
        c2 = gas(:v, :consvar).sound_speed2(vol, consvar)
        A[k] = c2 / mk * (J * tau / vol)^2
    end

    # diagonal coefficient B[l] and reduced residual R[l]
    let ml_g2 = (gravity^-2) * ml[1]
        B[1] = A[1] + ml_g2 + tau^2 * J * rhob # includes spring BC
        R[1] = ml_g2 * rPhi[1] + tau * rW[1]
    end
    for l in 2:Nz
        ml_g2 = (gravity^-2) * ml[l]
        B[l] = (A[l] + A[l - 1]) + ml_g2
        R[l] = ml_g2 * rPhi[l] + tau * rW[l]
    end
    let ml_g2 = (gravity^-2) * ml[Nz + 1]
        B[Nz + 1] = A[Nz] + ml_g2
        R[Nz + 1] = ml_g2 * rPhi[Nz + 1] + tau * rW[Nz + 1]
    end
    return A, B, R
end

function Thomas(A, B, R)
    # Thomas algorithm for symmetric tridiagonal system
    Nz = length(A)
    C, D, x = similar(A), similar(B), similar(R)
    # Forward sweep
    let l = 1
        X = inv(B[l])
        C[l] = -A[l] * X
        D[l] = R[l] * X
    end
    for l in 2:Nz
        X = inv(B[l] + A[l - 1] * C[l - 1])
        D[l] = (R[l] + A[l - 1] * D[l - 1]) * X
        C[l] = -A[l] * X
    end
    let l = Nz + 1
        X = inv(B[l] + A[l - 1] * C[l - 1])
        D[l] = (R[l] + A[l - 1] * D[l - 1]) * X
    end
    # Back-substitution
    x[Nz + 1] = D[Nz + 1]
    for l in Nz:-1:1
        x[l] = D[l] - C[l] * x[l + 1]
    end
    return x
end

function Thomas_flip(A, B, R, fl)
    # Thomas algorithm for symmetric tridiagonal system
    Nz = length(A)
    C, D, x = similar(A), similar(B), similar(R)
    function indices(l)
        flip(l) = fl ? (2+Nz-l, 1+Nz-l) : (l, l)
        prev(kl) = fl ? kl+1 : kl-1
        ln, kn = flip(l)
        return ln, prev(ln), kn, prev(kn)
    end
    # Forward sweep
    let (ln, lp, kn, kp) = indices(1)        
        X = inv(B[ln])
        C[kn] = -A[kn] * X
        D[ln] = R[ln] * X
#        @info "ln=$ln lp=$lp kn=$kn kp=$kp" X A[kn] B[ln] C[kn] D[ln]
    end
    for l in 2:Nz
        ln, lp, kn, kp = indices(l)
        X = inv(B[ln] + A[kp] * C[kp])
        D[ln] = (R[ln] + A[kp] * D[lp]) * X
        C[kn] = -A[kn] * X
#        @info "ln=$ln lp=$lp kn=$kn kp=$kp" X A[kn] B[ln] C[kn] D[ln]
    end
    let (ln, lp, kn, kp) = indices(Nz+1)        
        X = inv(B[ln] + A[kp] * C[kp])
        D[ln] = (R[ln] + A[kp] * D[lp]) * X
        # Back-substitution
        x[ln] = D[ln]
#        @info "ln=$ln lp=$lp kn=$kn kp=$kp" X B[ln] D[ln]
    end
    for l in Nz+1:-1:2
        ln, lp, kn, kp = indices(l)
        x[lp] = D[lp] - C[kp] * x[ln]
#        @info "ln=$ln lp=$lp kn=$kn kp=$kp" x[lp]
    end
    return x
end

function check_Thomas(A, B, R, x)
    # linear system is:
    # -A[l-1]*x[l-1] + B[l]*x[l] - A[l]*x[l+1] = R[l]
    Nz = length(A)
    R = copy(R)
    R[1] -= B[1] * x[1] - A[1] * x[2]
    for l in 2:Nz
        R[l] -= B[l] * x[l] - (A[l - 1] * x[l - 1] + A[l] * x[l + 1])
    end
    R[Nz + 1] -= B[Nz + 1] * x[Nz + 1] - A[Nz] * x[Nz]
    return R # should be zero !
end

function trisolve(A,B,R, fl)
    flip(x) = fl ? x[length(x):-1:1] : x
    TD = SymTridiagonal(flip(B), flip(-A))
    return flip(TD\flip(R))
end
