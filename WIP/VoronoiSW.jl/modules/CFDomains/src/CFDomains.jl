module CFDomains

using ManagedLoops: ManagedLoops, @loops, @vec

# Domain types
export AbstractDomain, SpectralDomain, SpectralSphere, FDDomain
"""
Parent type of [`SpectralDomain`](@ref) and [`FDDomain`](@ref)
"""
abstract type AbstractDomain end
"""
    SpectralDomain <: AbstractDomain
parent type of [`SpectralSphere`](@ref)
"""
abstract type SpectralDomain <: AbstractDomain end
"""
    FDDomain <: AbstractDomain`
Parent type of [`Box2D`](@ref) and [`Box3D`](@ref)
"""
abstract type FDDomain <: AbstractDomain end

@inline interior(x::AbstractDomain) = interior(typeof(x))
@inline interior(domain::Type) = domain

#=============== All domains ================#

# For initialization and plotting purposes, [c]vectorfield[!] converts between vector and cvector

export grid
export dotprod_cvector, allocate_field, allocate_fields # allocate_cvector, allocate_scalar
export sample_scalar, sample_scalar!, sample_cvector, sample_cvector!
export vectorfield, vectorfield!, cvectorfield, cvectorfield!

@inline allocate_fields(syms::NamedTuple, domain::AbstractDomain, F::Type) = map(sym->allocate_field(Val(sym), domain, F), syms)
@inline allocate_fields(syms::Tuple, domain::AbstractDomain, F::Type) = Tuple( allocate_field(Val(sym), domain, F) for sym in syms )
@inline allocate_fields(syms::Tuple, domain::AbstractDomain, F::Type, backend) = Tuple( allocate_field(Val(sym), domain, F, backend) for sym in syms )
@inline allocate_field(sym::Symbol, domain::AbstractDomain, F::Type) = allocate_field(Val(sym), domain, F)
@inline allocate_field(sym::Symbol, nq::Int, domain::AbstractDomain, F::Type) = allocate_field(Val(sym), nq, domain, F)
@inline allocate_field(sym::Symbol, domain::AbstractDomain, F::Type, backend) = allocate_field(Val(sym), domain, F, backend)
@inline allocate_field(sym::Symbol, nq::Int, domain::AbstractDomain, F::Type, backend) = allocate_field(Val(sym), nq, domain, F, backend)
# @inline allocate_field(::Val{:scalar},  args...) = allocate_scalar(args...)
# @inline allocate_field(::Val{:cvector}, args...) = allocate_cvector(args...)
# @inline allocate_field(::Val{:vector},  args...) = allocate_vector(args...)

# dot product between vectors represented as complex numbers
dotprod_cvector(a,b) = real(conj(a)*b)

meshgrid(ai, bj) = [a for a in ai, b in bj], [b for a in ai, b in bj]

"""
    periodize!(data, box::AbstractBox, backend)
Enforce horizontally-periodic boundary conditions on array `data` representing
grid point values in `box`. `data` may also be a collection, in which case
`periodize!` is applied to each element of the collection. Call `periodize!`
on data obtained by computations involving horizontal averaging/differencing.
"""
@inline periodize!(datas::Tuple, box::AbstractDomain, args...) = periodize_tuple!(datas, box, args...)
function periodize_tuple!(datas::Tuple, box, args...)
    for data in datas
        periodize!(data, box, args...)
    end
    return datas
end

macro fast(code)
    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"]!="")
    return debug ? esc(code) : esc(quote @inbounds $code end)
end

#===================== Shell =====================#

struct Shell{nz, Layer<:AbstractDomain} <: AbstractDomain
    layer::Layer
end
Shell(sphere::M, nz) where M = Shell{nz,M}(sphere)

@inline Base.eltype(shell::Shell) = eltype(shell.layer)

@inline Layer(domain::Shell) = domain.layer
@inline layers(domain::Shell) = domain
@inline nlayer(domain::Shell{nz}) where nz = nz
@inline interfaces(domain::Shell{nz,M}) where {nz,M} = Shell{nz+1,M}(domain.layer)
@inline interior(data, domain::Shell) = data

allocate_field(val::Val, domain::Shell{nz}, F) where nz = allocate_shell(val, domain.layer, nz, F)
allocate_field(val::Val, nq::Int, domain::Shell{nz}, F) where nz = allocate_shell(val, domain.layer, nz, nq, F)
allocate_field(val::Val, domain::Shell{nz}, F, backend) where nz = allocate_shell(val, domain.layer, nz, F, backend)
allocate_field(val::Val, nq::Int, domain::Shell{nz}, F, backend) where nz = allocate_shell(val, domain.layer, nz, nq, F, backend)

# belongs to ManagedLoops
array(T, ::Union{Nothing, ManagedLoops.HostManager}, size...) = Array{T}(undef, size...)
# array(T, backend::Loops.DeviceBackend, size...) = Loops.to_device(Array{T}(undef, size...), backend)
# array(T, backend::Loops.WrapperBackend, size...) = array(T, backend.backend, size...)

@inline ijk( ::Type{Shell{nz, M}}, ij, k) where {nz, M} = (ij-1)*nz + k
@inline kplus( ::Type{Shell{nz, M}})      where {nz, M} = 1

@inline primal(domain::Shell{nz}) where nz = Shell(primal(domain.layer), nz)

#============= Spherical harmonics on the unit sphere ==========#
"Parent type for spherical domains using spherical harmonics."
abstract type SpectralSphere <: SpectralDomain end

#===================== Spherical Voronoi mesh =================#

abstract type UnstructuredDomain <: AbstractDomain end


struct SubMesh{sym, Dom<:UnstructuredDomain} <: UnstructuredDomain
    domain::Dom
end
@inline SubMesh(sym::Symbol, dom::D) where D = SubMesh{sym,D}(dom)

@inline interior(submesh::SubMesh{sym}) where sym = interior(Val(sym), submesh.domain)

include("VoronoiSphere.jl")

end # module
