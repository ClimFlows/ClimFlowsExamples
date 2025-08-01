module Dynamics

using MutatingOrNot: void, Void, similar!
using ManagedLoops: @with, @vec

using SHTnsSpheres: SHTnsSphere, 
    analysis_scalar!, analysis_vector!,
    synthesis_scalar!, synthesis_vector!, synthesis_spheroidal!

using ..CFCompressible: FCE
import ..CFCompressible: FCE_tendencies!

#= Units
[m] = kg
[w] = s             w = g⁻²̇Φ
[W] = kg⋅s             
[p] = kg⋅m⁻¹⋅s⁻²
[ρ] = kg⋅m⁻³
[J] = m⋅s²         J = a²/g
[Jp] = kg
=#

#=
Computation of tendencies is split into the following steps:
1- Evaluate spatial inputs of HEVI solver: Phiₗ, Wₗ, mₖ, mₗ, sₖ
2- HEVI solver => new spatial values of W, Phi
3- fast tendencies for W, Phi
4- fast tendencies fur u,v
5- new spectral values for u,v
6- slow spectral tendencies for masses (mass budgets) and W, Phi (advection)
7- slow spectral tendencies for u,v (curl form)

Each step has its own additional scratch space for intermediate fields.
In addition, there is shared scratch space for (Phiₗ, Wₗ, mₖ, Sₖ, mₗ, sₖ)
=#

erase(x) = x # use SHTnsSpheres.erase when everything works

model_state(mass_air_spec, mass_consvar_spec, uv_spec, Phi_spec, W_spec) = 
    (; mass_air_spec, mass_consvar_spec, uv_spec, Phi_spec, W_spec)

function FCE_tendencies!(slow, fast, scratch, model, sph::SHTnsSphere, state, tau)
    # step 1
    common = spatial_fields!(scratch.common, model, sph, state)
    # step 2 : for the moment assume tau==0 and skip HEVI solver
    @assert tau==0
    # step 3
    fast_spat = fast_tendencies_PhiW!(scratch.fast_spat, model, common)
    dW_spec = analysis_scalar!(fast.W_spec, erase(fast_spat.dWl), sph)
    dPhi_spec = analysis_scalar!(fast.Phi_spec, erase(fast_spat.dPhil), sph)
    # step 4
    duv_spec, fast_uv = fast_tendencies_uv!(fast.uv_spec, scratch.fast_uv, model, sph, common.sk, fast_spat.dHdm, fast_spat.dHdS)
    zero_mass = ZeroArray(state.mass_air_spec)
    fast = model_state(zero_mass, zero_mass, duv_spec, dPhi_spec, dW_spec)

    # step 5: for the moment assume tau==0 and skip
    @assert tau==0
    new_state = state
    # step 6
    (dmass_air_spec, dmass_consvar_spec, dW_spec, dPhi_spec), slow_mass = mass_budgets!(slow, scratch.slow_mass, model, sph, new_state, common)
    # step 7
    duv_spec, slow_curl_form = curl_form!(slow.uv_spec, scratch.slow_curl_form, model.fcov, sph, new_state, slow_mass.fluxes, common.mk)
    # Done
    slow = model_state(dmass_air_spec, dmass_consvar_spec, duv_spec, dW_spec, dPhi_spec)
    scratch = (; common, fast_spat, fast_uv, slow_mass, slow_curl_form)
    return slow, fast, scratch
end

function spatial_fields!(scratch, model, sph, state)
    (; mass_air_spec, mass_consvar_spec, Phi_spec, W_spec) = state

    Phil = synthesis_scalar!(scratch.Phil, Phi_spec, sph)
    Wl = synthesis_scalar!(scratch.Wl, W_spec, sph)
    mk = synthesis_scalar!(scratch.mk, mass_air_spec, sph)
    Sk = synthesis_scalar!(scratch.Sk, mass_consvar_spec, sph)
    sk = similar!(scratch.sk, Sk)
    ml = similar!(scratch.ml, Wl)

    @with model.mgr let (irange, jrange) = (axes(mk,1), axes(mk, 2))
        Nz = size(mk, 3)
        for j in jrange, l in 1:(Nz + 1)
            @vec for i in irange
                if l == 1
                    mm = mk[i,j,1] / 2
                elseif l == Nz + 1
                    mm = mk[i,j,Nz] / 2
                else
                    mm = (mk[i,j,l - 1] + mk[i,j,l]) / 2
                end
                ml[i,j,l] = mm
            end
        end
        for j in jrange, k in 1:Nz
            @vec for i in irange
                sk[i,j,k] = Sk[i,j,k] / mk[i,j,k]
            end
        end
    end # @with

    return (; Phil, Wl, mk, Sk, sk, ml)
end

#============= fast tendencies ================#

zero!(x) = @. x=0

function fast_tendencies_PhiW!(scratch, model, common)
    (; Phis, rhob, pb) = model # bottom boundary condition p = pb - rhob(Phi-Phis)
    (; vcoord, planet, gas) = model
    (; Phil, Wl, mk, sk, ml) = common
    
    dWl = similar!(scratch.dWl, Wl) # = -dHdPhi
    dPhil = similar!(scratch.dPhil, Phil) # =+dHdW
    dHdm = similar!(scratch.dHdm, mk)
    dHdS = similar!(scratch.dHdS, sk)

    ptop, grav2, J = vcoord.ptop, planet.gravity^2, planet.radius^2/planet.gravity
       
    foreach(zero!, (dWl, dPhil, dHdm, dHdS)) # to be optimized... later
    @with model.mgr let (irange, jrange) = (axes(mk,1), axes(mk, 2))
        Nz = size(mk, 3)
        for j in jrange, l in 1:(Nz + 1)
            @vec for i in irange
                # kinetic
                wm = Wl[i,j,l] / ml[i,j,l]
                dPhil[i,j,l] = grav2 * wm
                l > 1 && (dHdm[i,j,l - 1] -= grav2 * wm^2/4)
                l <= Nz && (dHdm[i,j,l] -= grav2 * wm^2/4)
            end
        end
        for j in jrange, k in 1:Nz
            @vec for i in irange
                # potential
                dHdm[i,j,k] += (Phil[i,j,k+1] + Phil[i,j,k]) / 2
                dWl[i,j,k] -= mk[i,j,k] / 2
                dWl[i,j,k+1] -= mk[i,j,k] / 2
                # internal
                s = sk[i,j,k]
                vol = J * (Phil[i,j,k+1] - Phil[i,j,k]) / mk[i,j,k]
                p = gas(:v, :consvar).pressure(vol, s)
                Jp = J * p
                dWl[i,j,k] -= Jp
                dWl[i,j,k+1] += Jp
                h, _, exner = gas(:p, :consvar).exner_functions(p, s)
                dHdm[i,j,k] += h - s * exner
                dHdS[i,j,k] += exner
            end
        end
        # boundary
        for j in jrange
            @vec for i in irange
                dWl[i,j,Nz+1] -= J * ptop
                dWl[i,j,1] -= J * (rhob * (Phil[i,j,1] - Phis[i,j]) - pb[i,j])
            end
        end
    end # @with

    return (; dPhil, dWl, dHdm, dHdS)
end

function fast_tendencies_uv!(duv_spec, scratch, model, sph, sk, dHdm, dHdS)
    (; fx, fy, s_gradT_spec, dHdS_spec, grad_dHdS, dHdm_spec) = scratch

    dHdS_spec = analysis_scalar!(dHdS_spec, erase(dHdS), sph)
    (ux, uy) = grad_dHdS = synthesis_spheroidal!(grad_dHdS, dHdS_spec, sph)
    fx = @. fx = sk * ux
    fy = @. fy = sk * uy
    s_gradT_spec = analysis_vector!(s_gradT_spec, erase(vector_spat(fx,fy)), sph)

    dHdm_spec = analysis_scalar!(dHdm_spec, erase(dHdm), sph) # FIXME: avoid allocation
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = -dHdm_spec-s_gradT_spec.spheroidal),
        (@. duv_spec.toroidal = -s_gradT_spec.spheroidal),
    )

    return duv_spec, (; fx, fy, s_gradT_spec, dHdS_spec, grad_dHdS, dHdm_spec)
end

#============= slow tendencies ================#

function mass_budgets!(dstate, scratch, model, sph, state, common)
    (; mgr, planet), (; laplace) = model, sph      # parameters
    (; mk, ml, sk), (; Phi_spec, uv_spec) = common, state   # inputs

    Wl = synthesis_scalar!(scratch.Wl, state.W_spec, sph)
    (vx, vy) = uv = synthesis_vector!(scratch.uv, uv_spec, sph)
    (gx, gy) = grad_Phi = synthesis_spheroidal!(scratch.grad_Phi, Phi_spec, sph)
    fluxes = NH_fluxes!(scratch.fluxes, mk, sk, vx, vy, gx, gy, Wl, ml, mgr, planet)

    # air mass budget
    flux_spec = analysis_vector!(scratch.flux_spec, vector_spat(fluxes.U, fluxes.V), sph)
    dmass_air_spec = @. dstate.mass_air_spec = flux_spec.spheroidal * laplace
    # consvar mass budget
    flux_spec = analysis_vector!(scratch.flux_spec, erase(vector_spat(fluxes.sU, fluxes.sV)), sph)
    dmass_consvar_spec = @. dstate.mass_consvar_spec = flux_spec.spheroidal * laplace
    # W budget
    Wflux_spec = analysis_vector!(scratch.Wflux_spec, erase(vector_spat(fluxes.wU, fluxes.wV)), sph)
    dW_spec = @. dstate.W_spec = Wflux_spec.spheroidal * laplace
    # Phi tendency
    dPhi_spec = analysis_scalar!(dstate.Phi_spec, fluxes.dPhi, sph)

    return (dmass_air_spec, dmass_consvar_spec, dW_spec, dPhi_spec), (; Wl, uv, grad_Phi, fluxes, flux_spec, Wflux_spec)
end

function NH_fluxes!(scratch, mk, sk, vx, vy, gx, gy, Wl, ml, mgr, planet)
    # covariant momentum (u,v) => contravariant mass flux (U,V)
    B = similar!(scratch.B, mk)
    U = similar!(scratch.U, vx)
    V = similar!(scratch.V, vy)
    sU = similar!(scratch.sU, vx)
    sV = similar!(scratch.sV, vy)
    wU = similar!(scratch.wU, Wl)
    wV = similar!(scratch.wU, Wl)
    dPhi = similar!(scratch.dPhi, Wl)

    factor = planet.radius^-2
    @with mgr let (irange, jrange) = (axes(mk,1), axes(mk, 2))
        Nz = size(mk, 3)
        # full levels
        for j in jrange, k in 1:Nz
            @vec for i in irange
                wl_d = Wl[i,j,k]/ml[i,j,k]
                wl_u = Wl[i,j,k+1]/ml[i,j,k+1]
                # U = a⁻² m (v - W/m ∇Φ), sU
                U[i,j,k] = factor*mk[i,j,k]*(vx[i,j,k]-(wl_d*gx[i,j,k]+wl_u*gx[i,j,k+1])/2)
                V[i,j,k] = factor*mk[i,j,k]*(vy[i,j,k]-(wl_d*gy[i,j,k]+wl_u*gy[i,j,k+1])/2)
                sU[i,j,k] = sk[i,j,k] * U[i,j,k]
                sV[i,j,k] = sk[i,j,k] * V[i,j,k]
            end
        end
        # interfaces
        for j in jrange, l in 1:Nz+1
            @vec for i in irange
                if l==1
                    Ul = U[i,j,l]/2
                    Vl = V[i,j,l]/2
                elseif l==Nz+1                    
                    Ul = U[i,j,l-1]/2
                    Vl = V[i,j,l-1]/2
                else
                    Ul = (U[i,j,l]+U[i,j,l-1])/2
                    Vl = (V[i,j,l]+V[i,j,l-1])/2
                end
                wl = Wl[i,j,l]/ml[i,j,l]
                wU[i,j,l], wV[i,j,l] = Ul * wl, Vl * wl # wU → ∂ₜW = -∇⋅(wU)
                dPhi[i,j,l] = -(Ul*gx[i,j,l]+Vl*gy[i,j,l])/ml[i,j,l] # ∂ₜΦ = -u⋅∇Φ
            end
        end
        # full levels again: Bernoulli function dH/dm
        for j in jrange, k in 1:Nz
            @vec for i in irange
                X_d = dPhi[i,j,k]*(Wl[i,j,k]/ml[i,j,k])  # (W/m) (-u⋅∇Φ) → m²⋅s⁻²
                X_u = dPhi[i,j,k+1]*(Wl[i,j,k+1]/ml[i,j,k+1])
                K = (U[i,j,k]^2+V[i,j,k]^2)/(2*factor*mk[i,j,k]^2) # a^2 u⋅u/2
                B[i,j,k] = K - (X_d+X_u)/2  # u⋅u/2 + (u⋅∇Φ)W/m
            end
        end
    end # @with
    return (; U, V, sU, sV, wU, wV, dPhi, B)
end

function curl_form!(duv_spec, scratch, fcov, sph, state, fluxes, mk)
    @info typeof(fluxes)
    (; U, V, B) = fluxes
    (; laplace) = sph
    zeta_spec = @. scratch.zeta_spec = -laplace * state.uv_spec.toroidal # curl
    zeta = synthesis_scalar!(scratch.zeta, zeta_spec, sph)
    qfx = @. scratch.qfx =  V * (zeta + fcov)/mk
    qfy = @. scratch.qfy =  -U * (zeta + fcov)/mk
    qflux_spec = analysis_vector!(scratch.qflux_spec, erase(vector_spat(qfx, qfy)), sph)

    B_spec = analysis_scalar!(scratch.B_spec, erase(B), sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )
    return duv_spec, (; zeta_spec, zeta, qfx, qfy, qflux_spec, B_spec)
end

#======================== utilities ====================#

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)

struct Zero <: Real end

struct ZeroArray{N} <: AbstractArray{Zero,N}
    ax::NTuple{N, Base.OneTo{Int}}
end
ZeroArray(x::AbstractArray) = ZeroArray(axes(x))
ZeroArray(x::NamedTuple) = map(ZeroArray, x)

@inline Base.axes(z::ZeroArray) = z.ax
@inline Base.getindex(::ZeroArray, i...) = Zero()

@inline Base.:*(::Number, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Number) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()

@inline Base.:+(x::Number, ::Zero) = x
@inline Base.:+(::Zero, x::Number) = x
@inline Base.:+(x::Complex, ::Zero) = x   # needed to disambiguate ::Number + ::Zero
@inline Base.:+(::Zero, x::Complex) = x   # needed to disambiguate ::Zero + ::Number
@inline Base.:+(::Zero, ::Zero) = Zero()

end # module
