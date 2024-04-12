using ManagedLoops: @loops, @vec, choose

wen1(qm, qp, U) = @vec if U > 0 qm else qp end

wen3(qmm, qm, qp, qpp, U) = @vec if U > 0 weno3(qmm, qm, qp) else weno3(qpp, qp, qm) end

wen5(qmmm, qmm, qm, qp, qpp, qppp, U) =
    @vec if U > 0 weno5(qmmm, qmm, qm, qp, qpp) else weno5(qppp, qpp, qp, qm, qmm) end

@inline weno_at_point(U, o, qmmm, qmm, qm, qp, qpp, qppp) =
    @vec if o >= 6 wen5(qmmm, qmm, qm, qp, qpp, qppp, U) else
    @vec if o >= 4 wen3(qmm, qm, qp, qpp, U) else
    @vec if o >= 2 wen1(qm, qp, U) else
    zero(qm) end end end

getrange(q, step) = 1+3*step:length(q)-2*step

@inline stencil6(q, i, step) =
    @inbounds (q[i-3*step], q[i-2*step], q[i-step], q[i], q[i+step], q[i+2*step])

@inline stencil4(q, i, step) = (q[i-2*step], q[i-step], q[i], q[i+step])

@inline stencil2(q, i, step) = (q[i-step], q[i])

function weno3(qm, q0, qp)
    eps = 1e-14

    qi1 = -qm / 2 + 3 * q0 / 2
    qi2 = (q0 + qp) / 2

    beta1 = (q0 - qm)^2
    beta2 = (qp - q0)^2
    tau = abs(beta2 - beta1)

    g1, g2 = 1/3, 2/3
    w1 = g1 * (1 + tau / (beta1 + eps))
    w2 = g2 * (1 + tau / (beta2 + eps))

    return (w1 * qi1 + w2 * qi2) / (w1 + w2)
end


function weno5(qmm, qm, q0, qp, qpp)
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    # factor 6 missing here
    qi1 = 2 * qmm - 7 * qm + 11 * q0
    qi2 = -qm + 5 * q0 + 2 * qp
    qi3 = 2 * q0 + 5 * qp - qpp

    k1, k2 = 13.0 / 12, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0)^2 + k2 * (qmm - 4 * qm + 3 * q0)^2
    beta2 = k1 * (qm - 2 * q0 + qp)^2 + k2 * (qm - qp)^2
    beta3 = k1 * (q0 - 2 * qp + qpp)^2 + k2 * (3 * q0 - 4 * qp + qpp)^2

    tau5 = abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1 + tau5 / (beta1 + eps))
    w2 = g2 * (1 + tau5 / (beta2 + eps))
    w3 = g3 * (1 + tau5 / (beta3 + eps))

    # factor 6 is hidden below
    return (w1 * qi1 + w2 * qi2 + w3 * qi3) / (6 * (w1 + w2 + w3))
end


@loops function wenoflux!(_, out, q, U, step, o)
    # reconstruction + multitplication with U[i] -> flux
    let irange = getrange(q, step)
        @inbounds @vec for i in irange
            out[i] = U[i] * weno_at_point(U[i], o[i], stencil6(q, i, step)...)
        end
    end
end

@loops function wenoreconstd!(_, out, q, U, step, o)
    # pure reconstruction, no multitplication with U[i]
    let irange = getrange(q, step)
        @inbounds @vec for i in irange
            out[i] = weno_at_point(U[i], o[i], stencil6(q, i, step)...)
        end
    end
end

@loops function add_divflux_1d!(_, dq, flx, msk, step)
    let irange = getrange(dq, step)
        @inbounds @vec for i in irange
            dq[i] += msk[i] * (flx[i] - flx[i+step])
        end
    end
end

function add_div_flux!(dq, flx, msk, step)
    n = length(dq)
    for i = 1:n-step
        dq[i] += (flx[i] - flx[i+step]) * msk[i]
    end
end


function diUn!(mgr, dq, q, U, V, msk, ox, oy, flx)
    nx, _ = size(q)
    for (u, order, step) in [(U, ox, 1), (V, oy, nx)]
        fill!(flx, 0)
        wenoflux!(mgr, flx, q, u, step, order)
        add_divflux_1d!(mgr, dq, flx, msk, step)
    end
end
