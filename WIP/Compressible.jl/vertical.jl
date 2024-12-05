struct VerticalEnergy{Gas, F}
    gas::Gas
    gravity::F
    J::F  # Jacobian L/g, or just L if mass incorporates g
    ptop::F # pressure at domain top
    Phis::F # surface geopotential
    pb::F   # equilibrium bottom pressure, for non-rigid ground
    rhob::F # stiffness of bottom BC: p_bot = pb + rhob*(Phi-Phis)
end

function boundary_energy(v::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop) = v
    return J*((rhob/2)*(Phis[1]-Phib)^2 - pb*Phi[1] + ptop*Phi[end])
end

function internal_energy(v::VerticalEnergy, Phi, W, m, S)
    Nz=length(m)
    (; gas, J) = v
    return sum(1:Nz) do k
        vol = J*(Phi[k+1]-Phi[k])/m[k]
        gas(:v,:consvar).internal_energy(vol, S[k]/m[k])*m[k]
    end
end

function potential_energy(::VerticalEnergy, Phi, W, m, S)
    Nz=length(m)
    return sum(1:Nz) do k
        (Phi[k+1]+Phi[k])*(m[k]/2)
    end
end

function kinetic_energy(v::VerticalEnergy, Phi, W, m, S)
    Nz=length(m)
    (; gravity) = v
    # W is extensive, on the dual mesh (interfaces)
    return sum(1:Nz+1) do k
        if k==1 
            mm = m[1]/2
        elseif k==N+1
            mm = m[N]/2
        else
            mm = (m[k-1]+m[k])/2
        end
        w = gravity*W[k]/mm
        return mm*w^2/2
    end
end
