struct NewtonSolve
    niter::Int # number of Newton-Raphson iterations
    flip_solve::Bool # direction of LU solver passes: false => bottom-up then top-down ; true => top-down then bottom-up
    update_W::Bool # update W during Newton iteration (true), or only at the end (false)
    verbose::Bool
end

function NewtonSolve(user)
    def = (niter=5, flip_solve=false, update_W=false, verbose=false)
    options = merge(def, user)
    return NewtonSolve(options.niter, options.flip_solve, options.update_W, options.verbose)
end

function fwd_Euler(H, tau, state)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, state...)
    Phitau = @. Phi + tau * dHdW
    Wtau = @. W - tau * dHdPhi
    return Phitau, Wtau
end

function bwd_Euler(H::VerticalEnergy, newton::NewtonSolve, tau, state)
    (; gravity) = H
    (; niter, flip_solve, update_W, verbose) = newton

    inv_tau_g2 = inv(tau * gravity^2)
    (Phi_star, W_star, m, S) = state
    DPhi, Phi, W = zero(Phi_star), copy(Phi_star), copy(W_star)
    for iter in 1:niter
        rPhi, rW = residual(H, tau, (Phi, W, m, S), Phi_star, W_star)
        A, B, R = tridiag_problem(H, tau, Phi, W, m, S, rPhi, rW)
        dPhi = Thomas(A, B, R, flip_solve)
        @. DPhi += dPhi
        @. Phi = Phi_star + DPhi

        (iter==1 && verbose ) && @info "Initial residuals" L2(rPhi) L2(rW)

        if update_W || iter==niter
            # although the reduced residual does not depend on W analytically,
            # updating W during the iteration improves the final residual and is cheap
            # W = ml * (Phi-Phi_star) / (tau*g^2)
            Nz = length(m)
            let l = 1, ml = m[l] / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
            for l in 2:Nz
                ml = (m[l - 1] + m[l]) / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
            let l = Nz + 1, ml = m[l - 1] / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
        end
    end

    if verbose
        rPhi, rW = residual(H, tau, (Phi, W, m, S), Phi_star, W_star)
        @info "Final residuals" L2(rPhi) L2(rW)
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

function tridiag_problem(H, tau, Phi, W, m, S, rPhi, rW)
    (; gas, gravity, rhob, J) = H
    Nz = length(m)
    ml, A, B, R = map(x -> fill(NaN, size(x)), (Phi, m, Phi, Phi))

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

Thomas(A, B, R, flip_solve) = flip_solve ? Thomas_flip(A,B,R) : Thomas_noflip(A,B,R)

function Thomas_noflip(A, B, R)
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

function Thomas_flip(A, B, R)
    # Thomas algorithm for symmetric tridiagonal system
    Nz = length(A)
    C, D, x = similar(A), similar(B), similar(R)
    # Forward sweep
    let ll = Nz + 1
        X = inv(B[ll])
        C[ll - 1] = -A[ll - 1] * X
        D[ll] = R[ll] * X
    end
    for ll in Nz:-1:2
        X = inv(B[ll] + A[ll] * C[ll])
        D[ll] = (R[ll] + A[ll] * D[ll + 1]) * X
        C[ll - 1] = -A[ll - 1] * X
    end
    let ll = 1
        X = inv(B[ll] + A[ll] * C[ll])
        D[ll] = (R[ll] + A[ll] * D[ll + 1]) * X
        # Back-substitution
        x[ll] = D[ll]
    end
    for ll in 1:Nz
        x[ll + 1] = D[ll + 1] - C[ll] * x[ll]
    end
    return x
end

#====================== checks ========================#

L2(x) = sqrt(sum(x->x^2, x))
mindiff(x) = minimum(k -> x[k + 1] - x[k], 1:(length(x) - 1))
check(x) =
    if isnan(L2(x)) || isinf(L2(x))
        @info x
        error()
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

function trisolve(A, B, R, fl)
    flip(x) = fl ? x[length(x):-1:1] : x
    TD = SymTridiagonal(flip(B), flip(-A))
    return flip(TD \ flip(R))
end
