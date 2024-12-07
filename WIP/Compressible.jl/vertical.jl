struct VerticalEnergy{Gas,F}
    gas::Gas
    gravity::F
    J::F  # Jacobian L/g, or just L if mass incorporates g
    ptop::F # pressure at domain top
    Phis::F # surface geopotential
    pb::F   # equilibrium bottom pressure, for non-rigid ground
    rhob::F # stiffness of bottom BC: p_bot = pb + rhob*(Phi-Phis)
end
# VerticalEnergy(model, gravity, Phis, pb, rhob) = VerticalEnergy(model.gas, gravity, model.planet.radius^2/gravity, model.vcoord.ptop, Phis, pb, rhob)
# J = 1 assumes that m incorporates gravity and is per unit mass. m has units of Pa
function VerticalEnergy(model, gravity, Phis, pb, rhob)
    return VerticalEnergy(model.gas, gravity, one(Phis), model.vcoord.ptop, Phis, pb, rhob)
end

function boundary_energy(H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop) = H
    return J * ((rhob / 2) * (Phi[1] - Phis)^2 - pb * Phi[1] + ptop * Phi[end])
end

function grad(::typeof(boundary_energy), H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    dHdPhi[end] = ptop
    dHdPhi[1] = J * rhob * (Phi[1] - Phis) - pb
    return dHdPhi, dHdW, dHdm, dHdS
end

function internal_energy(H::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    gas, J = H.gas(:v, :consvar), H.J
    return sum(1:Nz) do k
        vol = J * (Phi[k + 1] - Phi[k]) / m[k] # specific volume
        return m[k] * gas.specific_internal_energy(vol, S[k] / m[k])
    end
end

function grad(::typeof(internal_energy), H::VerticalEnergy, Phi, W, m, S)
    (; gas, J) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    for k in 1:Nz
        s = S[k] / m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / m[k]
        p = gas(:v, :consvar).pressure(vol, s)
        Jp = J * p
        dHdPhi[k] += Jp
        dHdPhi[k + 1] -= Jp
        h, _, exner = gas(:p, :consvar).exner_functions(p, s)
        dHdm[k] = h - s * exner
        dHdS[k] = exner
    end
    return dHdPhi, dHdW, dHdm, dHdS
end

function potential_energy(::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    return sum(1:Nz) do k
        return (Phi[k + 1] + Phi[k]) * (m[k] / 2)
    end
end

function grad(::typeof(potential_energy), H::VerticalEnergy, Phi, W, m, S)
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    for k in 1:Nz
        dHdm[k] = (Phi[k + 1] + Phi[k]) / 2
        dHdPhi[k] += m[k] / 2
        dHdPhi[k + 1] += m[k] / 2
    end
    return dHdPhi, dHdW, dHdm, dHdS
end

function kinetic_energy(H::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    (; gravity) = H
    # W is extensive, on the dual mesh (interfaces)
    return sum(1:(Nz + 1)) do l
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        w = gravity * W[l] / mm
        return mm * w^2 / 2
    end
end

function grad(::typeof(kinetic_energy), H::VerticalEnergy, Phi, W, m, S)
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    (; gravity) = H
    for l in 1:(Nz + 1)
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        gw = gravity^2 * W[l] / mm
        dHdW[l] = gw
        l > 1 && (dHdm[l - 1] += gw / 2)
        l <= Nz && (dHdm[l] += gw / 2)
    end
    return dHdPhi, dHdW, dHdm, dHdS
end

function total_energy(args...)
    return boundary_energy(args...) +
           potential_energy(args...) +
           internal_energy(args...) +
           kinetic_energy(args...)
end

function grad(::typeof(total_energy), H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop, gravity, gas) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    (; gravity) = H
    for l in 1:(Nz + 1)
        # kinetic
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        gw = gravity^2 * W[l] / mm
        dHdW[l] += gw
        l > 1 && (dHdm[l - 1] += gw / 2)
        l <= Nz && (dHdm[l] += gw / 2)
    end
    for k in 1:Nz
        # potential
        dHdm[k] += (Phi[k + 1] + Phi[k]) / 2
        dHdPhi[k] += m[k] / 2
        dHdPhi[k + 1] += m[k] / 2
        # internal
        s = S[k] / m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / m[k]
        p = gas(:v, :consvar).pressure(vol, s)
        Jp = J * p
        dHdPhi[k] += Jp
        dHdPhi[k + 1] -= Jp
        h, _, exner = gas(:p, :consvar).exner_functions(p, s)
        dHdm[k] += h - s * exner
        dHdS[k] += exner
    end
    # boundary
    dHdPhi[end] += J*ptop
    dHdPhi[1] += J * (rhob * (Phi[1] - Phis) - pb)

    return dHdPhi, dHdW, dHdm, dHdS
end

function initial(H::VerticalEnergy, case, vcoord)
    Nz = CFDomains.nlayer(vcoord)
    lon, lat = 0.0, 0.0
    ps, Phis = case(lon, lat)
    W = zeros(Nz + 1)
    Phi = similar(W)
    m = zeros(Nz)
    S = similar(m)
    for l in 1:(Nz + 1)
        p = CFDomains.pressure_level(2l - 2, ps, vcoord)
        Phi[l], _, _, _ = case(lon, lat, p)
    end
    for k in 1:Nz
        p_down = CFDomains.pressure_level(2k - 2, ps, vcoord)
        p_mid = CFDomains.pressure_level(2k - 1, ps, vcoord)
        p_up = CFDomains.pressure_level(2k, ps, vcoord)
        m[k] = p_down - p_up
        vol = (case(lon, lat, p_up)[1] - case(lon, lat, p_down)[1]) / m[k] # specific volume
        s = H.gas(:p, :v).conservative_variable(p_mid, vol)
        S[k] = s * m[k]
    end
    return Phi, W, m, S
end

function test_grad(fun, H, state)
    dE = grad(fun, H, state...)
    E(state...) = fun(H, state...)
    dE_ = Enzyme.gradient(Reverse, E, state...)
    for (i, (dHdX, dHdX_)) in enumerate(zip(dE, dE))
        @info fun dHdX ≈ dHdX_ # dHdX[1] dHdX_[1]
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
#    dH = Enzyme.autodiff(set_runtime_activity(Forward), Const(f), Const, 0.)
    dH = Enzyme.autodiff(set_runtime_activity(Reverse), Const(f), Active, Active(0.))
    return f(0), dH
end
