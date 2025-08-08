function test_grad(fun, H, state)
    dE = grad(fun, H, state...)
    E(state...) = fun(H, state...)
    dE_ = Enzyme.gradient(Reverse, E, state...)
    for (i, (dHdX, dHdX_)) in enumerate(zip(dE, dE))
        @info fun dHdX ≈ dHdX_
    end
end

function test_canonical(H, state)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, state...)
    function f(tau)
        Phitau = @. Phi - tau * dHdW
        Wtau = @. W + tau * dHdPhi
        return total_energy(H, Phitau, Wtau, m, S)
    end
    dH = Enzyme.autodiff(set_runtime_activity(Reverse), Const(f), Active, Active(0.0))
    return f(0), dH
end
