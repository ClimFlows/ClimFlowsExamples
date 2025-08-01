struct NewtonSolve
    niter::Int # number of Newton-Raphson iterations
    flip_solve::Bool # direction of LU solver passes: false => bottom-up then top-down ; true => top-down then bottom-up
    update_W::Bool # update W during Newton iteration (true), or only at the end (false)
    verbose::Bool
end

function NewtonSolve(; niter=5, flip_solve=false, update_W=false, verbose=false, other...)
    return NewtonSolve(niter, flip_solve, update_W, verbose)
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
        dPhi = Solvers.Thomas(A, B, R, flip_solve)
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

#====================== checks ========================#

L2(x) = sqrt(sum(x->x^2, x))
mindiff(x) = minimum(k -> x[k + 1] - x[k], 1:(length(x) - 1))
check(x) =
    if isnan(L2(x)) || isinf(L2(x))
        @info x
        error()
    end
