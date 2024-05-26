#================= multi-layer domain ==================#

struct HVLayout end # memory layout where horizontal layers are contiguous
struct VHLayout end # memory layout where columns are contiguous

struct Shell{nz, L, D}
    layout::L
    layer::D
    Shell(nz::Int, layout::L, layer::D) where {L,D} = new{nz, L, D}(layout, layer)
end
shell(layer::SHTnsSphere, nz) = Shell(nz, HVLayout(), layer)
shell(layer::VoronoiSphere, nz) = Shell(nz, VHLayout(), layer)

layers(::Shell{nz}) where nz = nz

#================= Lagrangian HPE =================#
"""
    abstract type VerticalCoordinate{N} end

Parent type for generalized vertical coordinates ranging from `0` to `N`.
See also [`PressureVerticalCoordinate`](@ref).
"""
abstract type VerticalCoordinate{N} end

"""
    abstract type PressureCoordinate{N} <: VerticalCoordinate{N} end

Parent type for a pressure-based vertical coordinate.
Children types should specialize [`pressure_level`](@ref).
See also [`VerticalCoordinate`](@ref).
"""
abstract type PressureCoordinate{N} <: VerticalCoordinate{N}
end

"""
    p = pressure_level(k, ps, vcoord::PressureCoordinate{N})

Given surface pressure `ps`, returns the value `p`
of pressure corresponding to level *`k/2`* for vertical coordinate vcoord.
This means that so-called full levels correspond to odd
values `k=1,2...2N-1` while interfaces between full levels
(so-called half-levels) correspond to even values `k=0,2...2N`
"""
function pressure_level end

"""
    sigma = SigmaCoordinate(N, ptop) <: PressureCoordinate{N}
Pressure based sigma-coordinate for `N` levels with top pressure `ptop`.
Pressure levels are linear in vertical coordinate `k` :
    k/N = (ps-p)/(ps-ptop)
where `k` ranges from `0` (ground) to `N` (model top).
"""
struct SigmaCoordinate{N,F} <: PressureCoordinate{N} 
    ptop:: F # pressure at model top
end
SigmaCoordinate(N::Int, ptop::F) where F = SigmaCoordinate{N,F}(ptop)

pressure_level(k, ps, sigma::SigmaCoordinate{N}) where N = ( k*sigma.ptop + (2N-k)*ps )/2N

struct LagrangianHPE{Manager, Domain, VCoord, Planet, Gas<:AbstractFluid, Phisurf, Fcov, F}
    manager::Manager
    domain::Domain
    # Dynamics & vertical coordinate
    vcoord::VCoord
    planet::Planet
    ptop::F
    # Thermodynamics
    gas::Gas
    # Surface geopotential
    phisurf::Phisurf
    # Coriolis factor multiplied by Jacobian
    fcov::Fcov
end

function initialize_LHPE(shell::Shell{nz, HVLayout}, model, fun_ps, fun_Phi, args...) where nz
    sanity_checks(model, model.vcoord)
    ulon, ulat = allocate_fields( (:scalar_spat, :scalar_spat), shell, eltype(model) )
    pressure, geopot = allocate_fields( (:scalar_spat, :scalar_spat), Domains.interfaces(shell), eltype(model) )
    domain, vcoord, ptop = shell.layer, model.vcoord, model.ptop
    # p,Phi at mass points
    let (irange, jrange) = model.backend(axes(domain.lon) ; domain, model, nz, fun_ps, fun_Phi, args, vcoord, pressure, geopot)
        for i in irange, j in jrange
            lon, lat = domain.lon[i,j], domain.lat[i,j]
            ps = fun_ps(lon, lat, args...)
            for k=0:nz
                pressure[i,j, k+1] = p = pressure_level(2k, ps, vcoord)
                geopot[i,j, k+1], _... = fun_Phi(lon, lat, p, args...)
                if k>0
                    p = pressure_level(2k-1, ps, vcoord)
                    _, ulon[i,j,k], ulat[i,j,k], _... = fun_Phi(lon, lat, p, args...)
                end
            end
            model.phisurf[i,j]=geopot[i,j,1] # surface geopotential
        end
    end
    return pressure, geopot, ulon, ulat
end

