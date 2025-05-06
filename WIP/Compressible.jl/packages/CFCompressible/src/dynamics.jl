module Dynamics

using ManagedLoops: @with, @vec
using SHTnsSpheres: SHTnsSphere
using ..CFCompressible: FCE
import ..CFCompressible: FCE_tendencies!

function FCE_tendencies!(slow, fast, scratch, model, sph::SHTnsSphere, state, tau)
    # fast dynamics
    # for the moment assume tau==0
    @assert tau==0
    fast, fast_spat, scratch_fast = fast_tendencies!(fast, scratch.fast_spat, scratch.fast, model, state)

    # epilogue
    dW_spec = analysis_scalar!(fast.W, erase(fast_spat.W), sph)
    dPhi_spec = analysis_scalar!(fast.Phi, erase(fast_spat.Phi), sph)
    duv_spec = analysis_vector!(fast.uv, erase(fast_spat.uv), sph)
    fast = (uv=duv_spec, Phi=dPhi_spec, W=dW_spec)
    scratch = (fast=scratch_fast, fast_spat)
    return slow, fast, scratch
end

function fast_tendencies!(fast_spat, sk_, scratch, model, state)
    (; Phis, vcoord, planet) = model
    (; mass_air_spec, mass_consvar_spec, uv_spec, Phi_spec, W_spec) = state

    Phil = synthesis_scalar!(scratch.Phil, Phi_spec, sph)
    Wl = synthesis_scalar!(scratch.Wl, W_spec, sph)
    mk = synthesis_scalar!(scratch.mk, mass_air_spec, sph)
    Sk = synthesis_scalar!(scratch.Sk, mass_consvar_spec, sph)

    sk = similar!(sk_, Sk)
    dWl = similar!(fast_spat.W, Wl) # = -dHdPhi
    dPhil = similar!(fast_spat.Phi, Phil) # =+dHdW
    dHdm = similar!(scratch.dHdm, mk)
    dHdS = similar!(scratch.dHdS, Sk)
    foreach(zero!, (dWl, dPhil, dHdm, dHdS)) # to be optimized... later

    ptop, grav2, J = vcoord.ptop, planet.gravity^2, planet.radius^2/planet.gravity
       
    @with model.mgr let (irange, jrange) = (axes(mk,1), axes(mk, 2))
        Nz = size(mk, 3)
        for j in jrange, l in 1:(Nz + 1)
            @vec for i in irange
                # kinetic
                if l == 1
                    mm = m[1] / 2
                elseif l == Nz + 1
                    mm = m[i,j,Nz] / 2
                else
                    mm = (m[i,j,l - 1] + m[i,j,l]) / 2
                end
                wm = W[i,j,l] / mm
                dPhil[i,j,l] = grav2 * wm
                l > 1 && (dHdm[i,j,l - 1] -= grav2 * wm^2/4)
                l <= Nz && (dHdm[i,j,l] -= grav2 * wm^2/4)
            end
        end
        for j in jrange, k in 1:Nz
            @vec for i in irange
                # potential
                dHdm[i,j,k] += (Phi[i,j,k+1] + Phi[i,j,k]) / 2
                dWl[i,j,k] -= m[i,j,k] / 2
                dWl[i,j,k+1] -= m[i,j,k] / 2
                # internal
                s = Sk[i,j,k] / mk[i,j,k]
                vol = J * (Phi[i,j,k+1] - Phi[i,j,k]) / m[i,j,k]
                p = gas(:v, :consvar).pressure(vol, s)
                Jp = J * p
                dWl[i,j,k] -= Jp
                dWl[i,j,k+1] += Jp
                h, _, exner = gas(:p, :consvar).exner_functions(p, s)
                dHdm[i,j,k] += h - s * exner
                dHdS[i,j,k] += exner
                sk[i,j,k] = s
            end
        end
        # boundary
        for j in jrange
            @vec for i in irange
                dWl[i,j,Nz+1] -= J * ptop
                dWl[i,j,1] -= J * (rhob * (Phi[i,j,1] - Phis) - pb)
            end
        end
    end # @with

    scratch = (; dHdm, dHdS, Phil, Wl, ml, Sl)
    fast_spat = (mass_air=0, mass_consvar=0, uv=duv_k, Phi=dPhil, W=dWl)
    return fast_spat, sk, scratch
end

end # module
