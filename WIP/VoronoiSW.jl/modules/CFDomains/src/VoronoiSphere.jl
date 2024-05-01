macro fields(expr)
    @assert expr.head == :(::)
    typ = expr.args[2]
    lines = [ :( $field :: $typ) for field in expr.args[1].args ]
    return esc(Expr(:block, lines... ))
end

struct VoronoiSphere{F,
    VI<:AbstractVector{Int32},  # Vectors of integers
    MI<:AbstractMatrix{Int32},  # Matrices of integers
    VR<:AbstractVector{F},      # Vectors of reals
    MR<:AbstractMatrix{F},      # Matrices of reals
    AR<:AbstractArray{F}       # 3D array of reals
    } <: UnstructuredDomain
    @fields ( primal_num, dual_num, edge_num ) :: Int32
    @fields ( primal_deg, dual_deg, trisk_deg ) :: VI
    @fields ( primal_edge, primal_vertex, dual_edge, dual_vertex ) :: MI
    @fields ( edge_left_right, edge_down_up, trisk ) :: MI
    @fields ( primal_ne, dual_ne ) :: MI
    @fields ( Ai, lon_i, lat_i, Av, lon_v, lat_v) :: VR
    @fields ( le, de, le_de, lon_e, lat_e, angle_e)    :: VR
    @fields ( primal_bounds_lon, primal_bounds_lat, dual_bounds_lon, dual_bounds_lat ) :: MR
    @fields ( Riv2, wee ) :: MR
    primal_perot_cov :: AR
end
const VSph = VoronoiSphere

Base.show(io::IO, ::Type{<:VoronoiSphere{F}}) where F = print(io,
    "VoronoiSphere{$F}")
Base.show(io::IO, sphere::VoronoiSphere) = print(io,
    "VoronoiSphere($(length(sphere.Ai)) cells, $(length(sphere.Av)) dual cells)")

# converts Floats to Float, leaves other types alone
@inline convert_float(data, T) = data
@inline convert_float(data::Integer, T) = data
@inline convert_float(data::AbstractFloat, T) = T(data)

function VoronoiSphere(read_data::Function ; prec=Float32)
    names = fieldnames(VoronoiSphere)
    data = Dict( name => convert_float.(read_data(name), prec) for name in names)
    nums = (:primal_num, :dual_num, :edge_num)
    for name in nums
        data[name] = Int32(data[name])
    end
    nums = Tuple( data[name] for name in nums )
    return VoronoiSphere((crop(nums, data[name], name) for name in names)...)
end

@inline Base.eltype(dom::VSph) = eltype(dom.Ai)

@inline primal(dom::VSph) = SubMesh{:scalar, typeof(dom)}(dom)
@inline interior(::Val{:scalar}, dom::VSph) = eachindex(dom.Ai)

function crop((primal_num, dual_num, edge_num), data, name::Symbol)
    if name in (:primal_deg, :primal_edge, :primal_vertex, :primal_ne,
        :Ai, :lon_i, :lat_i, :primal_bounds_lon, :primal_bounds_lat)
        num  = primal_num
    elseif name in (:dual_deg, :dual_edge, :dual_vertex, :dual_ne,
        :Av, :lon_v, :lat_v, :dual_bounds_lon, :dual_bounds_lat, :Riv2)
        num = dual_num
    elseif name in (:trisk_deg, :edge_left_right, :edge_down_up, :trisk, :le, :de, :le_de,
        :lon_e, :lat_e, :angle_e, :wee)
        num = edge_num
    elseif name == :primal_perot_cov
        return data[:,1:primal_num, :]
    else
        return data
    end
    if isa(data, AbstractVector)
        return data[1:num]
    else
        return data[:,1:num]
    end
end

#====================== Allocate ======================#

allocate_field(::Val{:scalar}, dom::VSph, F::Type{<:Real}, backend=nothing) = array(F, backend, length(dom.Ai))
allocate_field(::Val{:dual},   dom::VSph, F::Type{<:Real}, backend=nothing) = array(F, backend, length(dom.Av))
allocate_field(::Val{:vector}, dom::VSph, F::Type{<:Real}, backend=nothing) = array(F, backend, length(dom.le))

allocate_shell(::Val{:scalar}, dom::VSph, nz, F::Type, backend=nothing) = array(F, backend, nz, length(dom.Ai))
allocate_shell(::Val{:dual},   dom::VSph, nz, F::Type, backend=nothing) = array(F, backend, nz, length(dom.Av))
allocate_shell(::Val{:vector}, dom::VSph, nz, F::Type, backend=nothing) = array(F, backend, nz, length(dom.le))
allocate_shell(::Val{:scalar}, dom::VSph, nz, nq, F::Type, backend=nothing) = array(F, backend, nz, length(dom.Ai), nq)
allocate_shell(::Val{:dual},   dom::VSph, nz, nq, F::Type, backend=nothing) = array(F, backend, nz, length(dom.Av), nq)
allocate_shell(::Val{:vector}, dom::VSph, nz, nq, F::Type, backend=nothing) = array(F, backend, nz, length(dom.le), nq)

@inline periodize!(data, ::Shell{Nz,<:VSph}, backend) where Nz = data
@inline periodize!(data, ::Shell{Nz,<:VSph}) where Nz = data
@inline periodize!(datas::Tuple, ::Shell{Nz, <:VSph}, args...) where Nz = datas

#====================== Effective resolution ======================#

function grad!(gradcov::AbstractVector, f, left_right)
    @fast for ij in eachindex(gradcov)
        gradcov[ij] = f[left_right[2,ij]]-f[left_right[1,ij]]
    end
end

function div!(divu, ucov, degree, areas, edges, hodges, signs)
    @fast for ij in eachindex(divu)
        deg = degree[ij]
        divu[ij] = inv(areas[ij]) * sum( (signs[e,ij]*hodges[edges[e,ij]])*ucov[edges[e,ij]] for e=1:deg )
    end
end

normL2(f) = sqrt( sum(x*x for x in f)/length(f) )

"""
Estimates the largest eigenvalue `-lambda=dx^-2` of the scalar Laplace operator and returns `dx`
which is a (non-dimensional) length on the unit sphere characterizing the mesh resolution.
By design, the Courant number for the wave equation with unit wave speed solved with time step `dt` is `dt/dx`.
"""
function laplace_dx(mesh::VoronoiSphere)
    h = randn(eltype(mesh.Ai), length(mesh.Ai))
    u = similar(mesh.le_de)
    for i=1:20
        hmax = normL2(h)
        @. h = inv(hmax)*h
        gradient!(u, h, mesh)
        divergence!(h, u, mesh)
    end
    return inv(sqrt(normL2(h))) :: eltype(mesh.le_de)
end

gradient!(u, h, mesh::VoronoiSphere) = grad!(u,h, mesh.edge_left_right)
divergence!(h, u, mesh::VoronoiSphere) = div!(h, u, mesh.primal_deg, mesh.Ai, mesh.primal_edge, mesh.le_de, mesh.primal_ne)

#========================== Interpolation ===========================#

# First-order interpolation weighted by areas of dual cells
primal_from_dual!(fi,fv, mesh::VoronoiSphere) = primal_from_dual!(fi, fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

function primal_from_dual!(fi::AbstractVector, fv, degrees, areas, vertices)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        Ai = sum(areas[vertices[vertex,ij]] for vertex=1:deg)
        fi[ij] = inv(Ai)*sum(
            areas[vertices[vertex,ij]]*fv[vertices[vertex,ij]] for vertex=1:deg)
    end
    return fi
end

function primal_from_dual!(fi::AbstractMatrix, fv, degrees, areas, vertices)
    nz = size(fi,1)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        inv_Ai = inv(sum(areas[vertices[vertex,ij]] for vertex=1:deg))
        for k=1:nz fi[k,ij]=0 end
        for vertex = 1:deg
            vv = vertices[vertex,ij]
            ww = areas[vv]*inv_Ai
            for k=1:nz fi[k,ij] = muladd(ww, fv[k,vv], fi[k,ij]) end
        end
    end
    return fi
end

primal_from_dual(fv::AbstractVector, degrees, areas, vertices) = primal_from_dual!(
    similar(degrees, eltype(fv)), fv, degrees, areas, vertices)
primal_from_dual(fv::AbstractMatrix, degrees, areas, vertices) = primal_from_dual!(
    Matrix{eltype(fv)}(undef, size(fv,1), size(degrees,1)), fv, degrees, areas, vertices)
primal_from_dual(fv, mesh::VoronoiSphere) = primal_from_dual(fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

# Perot reconstruction of vector field given covariant components

function primal3D_from_cov!(u::T, v::T, w::T, ucov::T, degrees, edges, weights) where {T<:AbstractVector}
    for ij in eachindex(u, v, w, ucov, degrees)
        u[ij], v[ij], w[ij] = primal3D_from_cov!(ij, ucov, degree[ij], edges, weights)
    end
    return u, v, w
end

# x,y,z = ( coslat*coslon, coslat*sinlon, sinlat )
# d(x,y,z)/dlon  = ( -coslat*sinlon, coslat*coslon, 0 )
# d(x,y,w)/dlat  = ( -sinlat*coslon, -sinlat*coslon, coslat )
# function 'fun' is applied to (ulon,ulat), see ShallowWaters.diag_ulonlat
@inline function primal_lonlat_from_cov!(fun::Fun, ulon::T, ulat::T, ucov::T,
    degrees, edges, weights, coslon, sinlon, coslat, sinlat) where {Fun, T<:AbstractVector}
    for ij in eachindex(ulon, ulat, degrees, coslon, sinlon, coslat, sinlat)
        u, v, w = primal3D_from_cov!(ij, ucov, degrees[ij], edges, weights)
        ulon[ij], ulat[ij] = fun(ij, v*coslon[ij]-u*sinlon[ij], w*coslat[ij]-sinlat[ij]*(u*coslon[ij]+v*sinlon[ij]))
    end
    return ulon, ulat
end

@inline function primal_lonlat_from_cov!(fun::Fun, ulon::T, ulat::T, ucov::T,
    degrees, edges, weights, coslon, sinlon, coslat, sinlat) where {Fun, T<:AbstractMatrix}
    for ij in eachindex(degrees, coslon, sinlon, coslat, sinlat)
        for k in axes(ulon, 1)
            u, v, w = primal3D_from_cov!(ij, view(ucov, k, :), degrees[ij], edges, weights)
            ulon[k,ij], ulat[k,ij] = fun(ij, v*coslon[ij]-u*sinlon[ij], w*coslat[ij]-sinlat[ij]*(u*coslon[ij]+v*sinlon[ij]))
        end
    end
    return ulon, ulat
end

@inline primal3D_from_cov!(ij, ucov, deg, edges, weights) = (
    sum(weights[iedge,ij,1]*ucov[edges[iedge,ij]] for iedge=1:deg),
    sum(weights[iedge,ij,2]*ucov[edges[iedge,ij]] for iedge=1:deg),
    sum(weights[iedge,ij,3]*ucov[edges[iedge,ij]] for iedge=1:deg) )
