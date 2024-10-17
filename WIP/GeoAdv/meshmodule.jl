module MeshModule

using NetCDF
using NCDatasets
using LinearAlgebra

export Mesh, overview, create_mesh,
       MassField, mass_field, fill_field, add, locate_max, mass,
       DualField, dual_field,
       MassVector, mass_vector,
       MassVector3d, mass_vector3d,
       EdgeVector, edge_vector,
       MassVector, mass_vector, 
       Flux, flux, fill_flux, 
       NormalVector, normal_vector, fill_normal_vector,
       TangentialVector, tangential_vector, normal2tangential, 
       gradient, primal2dual, gradient_limiter!,
       speedreconst, meshconvert, divide,
       radialv, 
       create_netcdf_structure, add_time_slot, create_mass_field, put_mass_field

# Geometry on the sphere
function radialv(lat::Float64, lon::Float64)::Array{Float64}
    v = Array{Float64}(undef, 3)
    slat, clat = sincos(lat)
    slon, clon = sincos(lon)
    v[1] = clon * clat
    v[2] = slon * clat
    v[3] = slat
    return v
end

function normalize(v::Array{Float64})::Array{Float64}
    return (1 / norm(v,2)) * v
end

function arclen(v1::Array{Float64}, v2::Array{Float64})::Float64
    return asin(norm(cross(v1, v2)))
end

function triple_product(v1::Array{Float64}, v2::Array{Float64}, v3::Array{Float64})::Float64
    return dot(v1, cross(v2, v3))
end

#---------------------------------------------------------------------------------------------------
# Mesh structure and functions relative to it
struct Mesh
    # the order of the fields has no logic it follows the order of the fields
    # in ncdump
    fileName::String
    nx :: Int32
    ndual :: Int32
    nedge :: Int32
    max_primal_deg :: Int32
    max_dual_deg :: Int32
    lat_v::Array{Float64, 1}
    wee::Array{Float64, 2}
    dual_vertex::Array{Int32, 2}
    dual_edge::Array{Int32, 2}
    primal_bounds_lon::Array{Float64, 2}
    edge_down_up::Array{Int32, 2}
    primal_ne::Array{Int32, 2}
    Riv2::Array{Float64, 2}
    primal_perot::Array{Float64, 3}
    Ai::Array{Float64, 1}
    primal_edge::Array{Int32, 2}
    trisk_deg::Array{Int32, 1}
    Av::Array{Float64, 1}
    edge_left_right::Array{Int32, 2}
    lon_v::Array{Float64, 1}
    lon_e::Array{Float64, 1}
    trisk::Array{Int32, 2}
    primal_bounds_lat::Array{Float64, 2}
    primal_neighbour::Array{Int32, 2}
    primal_vertex::Array{Int32, 2}
    primal_gradv3d::Array{Float64, 3}
    dual_deg::Array{Int32, 1}
    lat_e::Array{Float64, 1}
    dual_bounds_lon::Array{Float64, 2}
    de::Array{Float64, 1}
    le::Array{Float64, 1}
    dual_ne::Array{Int32, 2}
    lat_i::Array{Float64, 1}
    angle_e::Array{Float64, 1}
    dual_bounds_lat::Array{Float64, 2}
    lon_i::Array{Float64, 1}
    primal_deg::Array{Int32, 1} 
    edge_kite::Array{Int32, 2}
    edge_perp::Array{Float64, 2}
    primal_grad3d::Array{Float64, 3} # Fields on this line and above are read from mesh.nc
    primal_nei::Array{Int32, 2} # Fields from this line are calculated by this module
    ulam_i::Array{Float64, 2}
    uphi_i::Array{Float64, 2}
    ulam_e::Array{Float64, 2}
    uphi_e::Array{Float64, 2}
    unit_out::Array{Float64, 3}
    normalvec_edge::Array{Float64, 2}
    tangvec_edge::Array{Float64, 2}
    radvec_i::Array{Float64, 2}
    radvec_e::Array{Float64, 2}
    radvec_v::Array{Float64, 2}
    cen2edge::Array{Float64, 3}
    x_cg::Array{Float64, 2}
    facets::Array{Int32, 2}
    facet_ne::Array{Int32, 2}
    dual_le::Array{Float64, 1}
    Ae::Array{Float64, 1}
end

function create_mesh(fileName::String)::Mesh
    # Create a mesh from a mesh.nc file. Include all the arrays in the netCDF as well as other
    # fields such as weights for gradient calculation, primal neighbours etc.
    
    lat_v = ncread(fileName, "lat_v")
    wee   = ncread(fileName, "wee")  
    dual_vertex = ncread(fileName, "dual_vertex")
    dual_edge = ncread(fileName, "dual_edge")  
    primal_bounds_lon = ncread(fileName, "primal_bounds_lon")
    edge_down_up = ncread(fileName, "edge_down_up")
    primal_ne = ncread(fileName, "primal_ne")
    Riv2 = ncread(fileName, "Riv2")
    primal_perot = ncread(fileName, "primal_perot")
    Ai = ncread(fileName, "Ai")
    primal_edge = ncread(fileName, "primal_edge")
    trisk_deg = ncread(fileName, "trisk_deg")
    Av = ncread(fileName, "Av")
    edge_left_right = ncread(fileName, "edge_left_right")
    lon_v = ncread(fileName, "lon_v") 
    lon_e = ncread(fileName, "lon_e") 
    trisk = ncread(fileName, "trisk") 
    primal_bounds_lat = ncread(fileName, "primal_bounds_lat")
    primal_neighbour = ncread(fileName, "primal_neighbour")
    primal_vertex = ncread(fileName, "primal_vertex")
    primal_gradv3d = ncread(fileName, "primal_gradv3d")
    dual_deg = ncread(fileName, "dual_deg")
    lat_e = ncread(fileName, "lat_e")
    dual_bounds_lon = ncread(fileName, "dual_bounds_lon")
    de = ncread(fileName, "de")
    le = ncread(fileName, "le")
    dual_ne = ncread(fileName, "dual_ne")
    lat_i = ncread(fileName, "lat_i")
    angle_e = ncread(fileName, "angle_e")
    dual_bounds_lat = ncread(fileName, "dual_bounds_lat")
    lon_i = ncread(fileName, "lon_i")
    primal_deg = ncread(fileName, "primal_deg")

    # Weights and stencils for gradient calculation an dspeed reconstruction
    edge_kite = ncread(fileName, "edge_kite")
    edge_perp = ncread(fileName, "edge_perp")
    primal_grad3d = ncread(fileName, "primal_grad3d")
    

    # Calculate things that we do not read directly from the netCDF
    nx    = size(lat_i)[1]        # this is primal_cell in the netCDF
    ndual = size(lat_v)[1]        # this is dual_cell in the netCDF
    nedge = size(lat_e)[1]        # this is edge in the netCDF
    max_primal_deg, index = findmax(primal_deg)
    max_dual_deg, index = findmax(dual_deg)

    # Create arrays that are not read in the NetCDF. We will fill them later.
    primal_nei = Array{Int32}(undef, max_primal_deg, nx)
    ulam_i     = Array{Float64}(undef, 3, nx)
    uphi_i     = Array{Float64}(undef, 3, nx)
    ulam_e     = Array{Float64}(undef, 3, nedge)
    uphi_e     = Array{Float64}(undef, 3, nedge)
    unit_out        = Array{Float64}(undef, 3, max_primal_deg, nx)
    normalvec_edge  = Array{Float64}(undef, 3, nedge)
    tangvec_edge  = Array{Float64}(undef, 3, nedge)
    radvec_i  = Array{Float64}(undef, 3, nx)
    radvec_e  = Array{Float64}(undef, 3, nedge)
    radvec_v  = Array{Float64}(undef, 3, ndual)
    cen2edge  = Array{Float64}(undef, 3, max_primal_deg, nx)
    x_cg       = Array{Float64}(undef, 3, nedge)
    facets     = Array{Int32}(undef, 4, nedge)
    facet_ne   = Array{Int32}(undef, 4, nedge)
    dual_le    = Array{Float64}(undef, nedge)
    Ae = Array{Float64}(undef, nedge)
    mesh = Mesh(fileName, nx, ndual, nedge, max_primal_deg, max_dual_deg,
        lat_v, wee, dual_vertex, dual_edge,
    primal_bounds_lon, edge_down_up, primal_ne, Riv2, 
    primal_perot, Ai, primal_edge, trisk_deg, Av, edge_left_right, lon_v, 
    lon_e, trisk, primal_bounds_lat, primal_neighbour, primal_vertex,
    primal_gradv3d, dual_deg, lat_e, dual_bounds_lon, de, le, 
    dual_ne, lat_i, angle_e, dual_bounds_lat, lon_i, primal_deg,
    edge_kite, edge_perp, primal_grad3d, 
    primal_nei, ulam_i, uphi_i, ulam_e, uphi_e, unit_out, normalvec_edge, tangvec_edge, 
    radvec_i, radvec_e, radvec_v, cen2edge,
    x_cg, facets, facet_ne, dual_le, Ae
    )

    unpack(mesh)

    return mesh
end # of function create_mesh

function obtain_neighbour(m::Mesh, edge_num, ix1)::Int32
    # Obtain the neighbour of ix1 across edge_num
    i1 = m.edge_left_right[1, edge_num]
    i2 = m.edge_left_right[2, edge_num]
    if i1 == ix1
       return i2
    else
       return i1
    end
end

function check_edge(m::Mesh, edge_num, ix1, ix2)::Bool
    # Returns true if edge edge_num separates primal cells ix1 and ix2
    i1 = m.edge_left_right[1, edge_num]
    i2 = m.edge_left_right[2, edge_num]
    if (i1 == ix1 && i2 == ix2) || (i1 == ix2 && i2 == ix1)
        return true
    else
        print(i1, " ",ix1, " ", i2, " ", ix2, " ", edge_num)
        return false
    end
end

function unpack(m::Mesh)
    # Precalculate arrays that are not included in mesh.nc but are useful for the rest of the
    # calculations:

    # 1. primal_nei[iedge, ix] : neighbour of ix across iedge
    # 2. ulam_i, uphi_i : eastward and northward unit vectors at the generators of primal mesh
    # 3. ulam_e, uphi_e : eastward and northward unit vectors at edge centers
    # 4. nolmalvec_edge and tangvec_edge, vectors normal and tangential to edges
    # 5. unit_out, outgoing normal vectors for each edge of each primal cell 
    # 6. length of dual edges
    # 7. Radial vectors to the generators, edges and vortices
    # 8. infinitesimal vectors from the generators to the edges
    
    # 1. primal_nei
    for ix in 1:m.nx
        for iedge in 1:m.primal_deg[ix]
            edge_num = m.primal_edge[iedge, ix]
            m.primal_nei[iedge, ix] = obtain_neighbour(m, edge_num, ix)
            verif = check_edge(m, edge_num, ix, m.primal_nei[iedge, ix])
            if !verif
                print("grid inconsistency found in meshmodule.jl\n")
                exit(1)
            end
        end
    end

    #2.
    # Calculate eastward and northward unit vectors in 3d on the primal mesh
    for ix in 1:m.nx
        m.ulam_i[1, ix] = - sin(m.lon_i[ix])
        m.ulam_i[2, ix] =   cos(m.lon_i[ix])
        m.ulam_i[3, ix] =   0.

        m.uphi_i[1, ix] = - sin(m.lat_i[ix]) * cos(m.lon_i[ix])  
        m.uphi_i[2, ix] = - sin(m.lat_i[ix]) * sin(m.lon_i[ix])  
        m.uphi_i[3, ix] =   cos(m.lat_i[ix])
    end
    
    #3.
    # Calculate eastward and northward unit vectors in 3d at edge centers
    for ix in 1:m.nedge
        m.ulam_e[1, ix] = - sin(m.lon_e[ix])
        m.ulam_e[2, ix] =   cos(m.lon_e[ix])
        m.ulam_e[3, ix] =   0.

        m.uphi_e[1, ix] = - sin(m.lat_e[ix]) * cos(m.lon_e[ix])  
        m.uphi_e[2, ix] = - sin(m.lat_e[ix]) * sin(m.lon_e[ix])  
        m.uphi_e[3, ix] =   cos(m.lat_e[ix])
    end

    #4.
    # Calculate vectors normal and tangential to edges
    for iedge in 1:m.nedge
        m.normalvec_edge[:,iedge] = ( cos(m.angle_e[iedge]) * m.ulam_e[:,iedge] + 
                                          sin(m.angle_e[iedge]) * m.uphi_e[:,iedge]
                                        )
        # tangvec is rotated counterclockwise by pi/2 so that (normalvec, tangvec, radial) is
        # always direct
        m.tangvec_edge[:,iedge] =   ( cos(m.angle_e[iedge] + pi / 2.) * m.ulam_e[:,iedge] + 
                                          sin(m.angle_e[iedge] + pi / 2.) * m.uphi_e[:,iedge]
                                        )
    end

    #5.
    # Calculate outgoing normal vectors
    m.unit_out .= 0.
    for ix in 1:m.nx
        for iedge in 1:m.primal_deg[ix]
            edge_num = m.primal_edge[iedge, ix]
            m.unit_out[:,iedge, ix] = m.primal_ne[iedge, ix] * m.normalvec_edge[:,edge_num] 
        end
    end
    print("checksum for unit_out", sum(m.unit_out, dims=(2,3)), "\n") 

    #6.
    # Construct the length of dual edges
    for iedge in 1:m.nedge
        c1 = m.edge_left_right[1,iedge]
        c2 = m.edge_left_right[2,iedge]
        m.dual_le[iedge] = arclen(
                                  radialv(m.lat_i[c1], m.lon_i[c1]),
                                  radialv(m.lat_i[c2], m.lon_i[c2])             
                                 )
    end

    #7.
    # calculate radial vectors
    for ix in 1:m.nx
        m.radvec_i[:,ix] .= radialv(m.lat_i[ix], m.lon_i[ix])
    end
    
    for ie in 1:m.nedge
        m.radvec_e[:,ie] .= radialv(m.lat_e[ie], m.lon_e[ie])
    end

    for iv in 1:m.ndual
        m.radvec_v[:,iv] .= radialv(m.lat_v[iv], m.lon_v[iv])
    end

    # 8.
    # Small vectors from the center to the edges
    for ix in 1:m.nx
        for iedge in 1:m.primal_deg[ix]
            edge_num = m.primal_edge[iedge, ix]
            m.cen2edge[:,iedge, ix] .= m.radvec_e[:,edge_num] .- m.radvec_i[:,ix] 
        end
    end
    
    return
end # of function unpack

function edgenum(m::Mesh, c1::Int32, c2::Int32)
    # returns the edge number that separates primal cells c1 and c2
    set2 = Set(m.primal_edge[1:m.primal_deg[c2],c2])
    i = -999
    for i in 1:m.primal_deg[c1]
        if m.primal_edge[i, c1] in set2
            return m.primal_edge[i, c1]
        end
    end
    print("ERROR: meshmodule.jl edgenum: common edge not found\n")
    exit(1)
end

function overview(m::Mesh)::String
    # Print some characteristics of the mesh.
    
    s = ""
    s = s * "mesh read from: " * m.fileName * "\n"
    s = s * "Primal cells: " * string(m.nx) * "\n"
    s = s * "Dual cells: " * string(m.ndual) * "\n"
    s = s * "Edges: " * string(m.nedge) * "\n"
    max_primal_deg , index = findmax(m.primal_deg)
    min_primal_deg , index = findmin(m.primal_deg)
    s = s * "min - max of primal degree: " *
            string(min_primal_deg) * " - " *
            string(max_primal_deg) * "\n"
    
    max_dual_deg , index = findmax(m.dual_deg)
    min_dual_deg , index = findmin(m.dual_deg)
    s = s * "min - max of dual degree: " *
            string(min_dual_deg) * " - " *
            string(max_dual_deg) * "\n"

    max , index = findmax(m.edge_left_right)
    min , index = findmin(m.edge_left_right)
    s = s * "min - max of edge_left_right: " *
            string(min) * " - " *
            string(max) * "\n"

    max , index = findmax(m.edge_down_up)
    min , index = findmin(m.edge_down_up)
    s = s * "min - max of edge_down_up: " *
            string(min) * " - " *
            string(max) * "\n"

    max , index = findmax(m.primal_ne)
    min , index = findmin(m.primal_ne)
    s = s * "min - max of primal_ne: " *
            string(min) * " - " *
            string(max) * "\n"
        
    max , index = findmax(m.angle_e)
    min , index = findmin(m.angle_e)
    s = s * "min - max of angle_e: " *
            string(min) * " - " *
            string(max) * "\n"

    max , index = findmax(m.Riv2)
    min , index = findmin(m.Riv2)
    s = s * "min - max of Riv2: " *
            string(min) * " - " *
            string(max) * "\n"
        
    max , index = findmax(sum(m.Riv2, dims=1))
    min , index = findmin(sum(m.Riv2, dims=1))
    s = s * "min - max of Riv2 sum: " *
            string(min) * " - " *
        string(max) * "\n"
        
    s = s * "Check sphere area: " * string(1. / (4pi) * sum(m.Ai)) * "\n"

    s = s * "Check losange area: " * string(1. / (12pi) * sum(m.Ae)) * "\n"

    return s

end # function overview for meshes
# Mesh structure and functions relative to it
#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------
# MassField structure and relative stuff

struct MassField
    nx::Int
    nz::Int
    F::Array{Float64}
end # struct MassField

function overview(m::MassField)::String
    # Print some properties of a MassField
    s = ""
    s = s * "This is a mass field with nx, nz: " * string(m.nx) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max of the field: " *
            string(minval) * " - " *
            string(maxval) * "\n"
    
    return s
end #function overview for massfields

function mass_field(m::Mesh, nz::Int)::MassField
    # Creates a mass field with UNDEF values
    F = Array{Float64}(undef, m.nx, nz)
    return MassField(m.nx, nz, F)
end # function mass_field

function fill_field(m::Mesh, field::MassField, f::Function)
    # Fill MassField with pointwise evaluation of function f at cell generators.
    # f is expected to have arguments: lat, lon, in radian.
    for ix in 1:m.nx
        val = f(m.lat_i[ix], m.lon_i[ix])
        for iz in 1:field.nz
            field.F[ix,iz] = val
        end
    end
    return
end

function fill_field(m::Mesh, field::MassField, val::AbstractFloat)
    # Fill MassField with a uniform value.
    for ix in 1:m.nx
        for iz in 1:field.nz
            field.F[ix,iz] = val
        end
    end
    return
end

function add(field1::MassField, field2::MassField)
    # Add field1 and field2 and store the result in field1
    for ix in 1:field1.nx
        for iz in 1:field1.nz
            field1.F[ix,iz] = field1.F[ix,iz] + field2.F[ix,iz]
        end
    end
end

function divide(field1::MassField, field2::MassField, result::MassField)
    # divide field1 by field2 and store the result in result
    for ix in 1:field1.nx
        for iz in 1:field1.nz
            result.F[ix,iz] = field1.F[ix,iz] / field2.F[ix,iz]
        end
    end
end

function locate_max(m::Mesh, field::MassField)
    # Give the geographic location and the value of the max of MassField
    max, index = findmax(field.F)
    lat = m.lat_i[index]
    lon = m.lon_i[index]
    return lat, lon, max
end

function mass(m::Mesh, field::MassField)
    # area-weighted integral of field, summed over the vertical
    s = 0.
    for iz in 1:field.nz
        s = s + sum(m.Ai .* field.F[:,iz])
    end
    return s
end

# MassField structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# DualField structure and relative stuff
struct DualField
    ndual::Int
    nz::Int
    F::Array{Float64}
end 

function overview(m::DualField)::String
    # Print some properties of a DualField
    s = ""
    s = s * "This is a dual field with ndual, nz: " * string(m.ndual) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max of the field: " *
            string(minval) * " - " *
            string(maxval) * "\n"
    
    return s
end

function dual_field(m::Mesh, nz::Int)::DualField
    # Creates a dual field with UNDEF values
    F = Array{Float64}(undef, m.ndual, nz)
    return DualField(m.ndual, nz, F)
end # function mass_field

function fill_field(m::Mesh, field::DualField, f::Function)
    # Fill DualField with pointwise evaluation of function f at cell generators.
    # f is expected to have arguments: lat, lon, in radian.
    for ix in 1:m.ndual
        val = f(m.lat_v[ix], m.lon_v[ix])
        for iz in 1:field.nz
            field.F[ix,iz] = val
        end
    end
    return
end

function fill_field(m::Mesh, field::DualField, val::AbstractFloat)
    # Fill DualField with a uniform value
    for ix in 1:m.ndual
        for iz in 1:field.nz
            field.F[ix,iz] = val
        end
    end
    return
end

function add(field1::DualField, field2::DualField)
    # Add field1 and field2 and store the results into field1
    for ix in 1:field1.ndual
        for iz in 1:field1.nz
            field1.F[ix,iz] = field1.F[ix,iz] + field2.F[ix,iz]
        end
    end
end
# DualField structure and relative stuff
#---------------------------------------------------------------------------------------------------

abstract type MeshField ; end

#---------------------------------------------------------------------------------------------------
# Flux structure and relative stuff
# A Flux is a float number on each edge, to be interpreted as a flux component across the edge, oriented 
# like normalvec_edge. It represents the integral of V.n along the edge where n is the unit vector 
# normal to the edge
struct Flux <: MeshField
    nedge::Int
    nz::Int
    F::Array{Float64}
end 
Base.getindex(fl::MeshField, elements...) = fl.F[elements...]
Base.setindex!(fl::MeshField, val, elements...) = setindex!(fl.F, val, elements...)

function overview(m::Flux)::String
    # Print some properties of a Flux
    s = ""
    s = s * "This is a flux with nedge, nz: " * string(m.nedge) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max of the field: " *
            string(minval) * " - " *
            string(maxval) * "\n"
    return s
end

function flux(m::Mesh, nz::Int)::Flux
    # Creates a Flux with UNDEF values
    F = Array{Float64}(undef, m.nedge, nz)
    return Flux(m.ndual, nz, F)
end # function fluc

function fill_flux(m::Mesh, fl::Flux, sf::DualField, factor = 1.)
    # Fill flux fl from streamfunction sf
    for iedge in 1:m.nedge
        vnum1    = m.edge_down_up[1, iedge]
        vnum2    = m.edge_down_up[2, iedge]
        val = sf.F[vnum2] - sf.F[vnum1]
        for iz in 1:fl.nz
            fl.F[iedge,iz] = val * factor
        end
    end
    return
end

function add(field1::Flux, field2::Flux)
    # Add field1 and field2 and store the results into field1
    for ix in 1:field1.nedge
        for iz in 1:field1.nz
            field1.F[ix,iz] = field1.F[ix,iz] + field2.F[ix,iz]
        end
    end
end
# Flux structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# NormalVector structure and relative stuff
# A NormalVector represents the component of a vector field normal to each edge, in the orientation of 
# normalvec_edge. A NormalVector multiplied by edge lengths gives a Flux. Inflrmatically, the structure 
# is the same as a Flux and with the same dimensions.
struct NormalVector
    nedge::Int
    nz::Int
    F::Array{Float64}
end 

function normal_vector(m::Mesh, nz::Int)::NormalVector
    # Creates a NormalVector with UNDEF values
    F = Array{Float64}(undef, m.nedge, nz)
    return NormalVector(m.nedge, nz, F)
end # function normal_vector

function fill_normal_vector(m::Mesh, nv::NormalVector, fl::Flux)
    # fills a NormalVector with values filled from a flux
    for iedge in 1:nv.nedge
        for iz in 1:nv.nz
            nv.F[iedge,iz] = fl.F[iedge,iz] / m.le[iedge]
        end
    end
    return nv
end # function normal_vector

function overview(m::NormalVector)::String
    # Print some properties of a Flux
    s = ""
    s = s * "This is a normal vector with nedge, nz: " * string(m.nedge) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max of the field: " *
             string(minval) * " - " *
             string(maxval) * "\n"
    return s
end

# NormalVector structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# TangentialVector structure and relative stuff
# A TangentialVector represents the component of a vector field tangential to each edge, in the orientation of 
# tangvec_edge. 
struct TangentialVector
    nedge::Int
    nz::Int
    F::Array{Float64}
end 

function tangential_vector(m::Mesh, nz::Int)::TangentialVector
    # Creates a TangentialVector with UNDEF values
    F = Array{Float64}(undef, m.nedge, nz)
    return TangentialVector(m.nedge, nz, F)
end # function normal_vector

function overview(m::TangentialVector)::String
    # Print some properties of a Tangential Vector
    s = ""
    s = s * "This is a tangential vector with nedge, nz: " * string(m.nedge) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max of the field: " *
            string(minval) * " - " *
            string(maxval) * "\n"
    return s
end

# TangentialVector structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# MassVector structure and relative stuff

struct MassVector
    nx::Int
    nz::Int
    u::Array{Float64}
    v::Array{Float64}
end # struct MassVector

function overview(m::MassVector)::String
    # Print some properties of a MassVector (e.g. gradient field)
    s = ""
    s = s * "This is a mass vector field with nx, nz: " * string(m.nx) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.u)
    maxval, index = findmax(m.u)
    s = s * "min - max of the zonal component: " *
            string(minval) * " - " *
        string(maxval) * "\n"
    minval, index = findmin(m.v)
    maxval, index = findmax(m.v)
    s = s * "min - max of the meridional component: " *
            string(minval) * " - " *
        string(maxval) * "\n"
   
    return s
    
end #function overview for MassVector

function mass_vector(m::Mesh, nz::Int)::MassVector
    # Creates a vector field on primal cell centers with UNDEF values
    u = Array{Float64}(undef, m.nx, nz)
    v = Array{Float64}(undef, m.nx, nz)
    return MassVector(m.nx, nz, u, v)
end # function mass_vector

function fill_field(m::Mesh, vector::MassVector, f::Function)
    # Fill MassVector with pointwise evaluation of function f at cell generators.
    # f is expected to return u and v from lat, lon in radians.
    for ix in 1:m.nx
        valu, valv = f(m.lat_i[ix], m.lon_i[ix])
        for iz in 1:vector.nz
            vector.u[ix,iz] = valu
            vector.v[ix,iz] = valv
        end
    end
    return
end

# MassVector structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# MassVector3d structure and relative stuff

struct MassVector3d
    nx::Int
    nz::Int
    F::Array{Float64}
end # struct MassVector3d

function overview(m::MassVector3d)::String
    # Print some properties of a MassVector (e.g. gradient field)
    s = ""
    s = s * "This is a mass vector 3d field with nx, nz: " * string(m.nx) * " " * string(m.nz) * "\n"
    minval, index = findmin(m.F)
    maxval, index = findmax(m.F)
    s = s * "min - max across the three components: " *
            string(minval) * " - " *
            string(maxval) * "\n"
    return s
    
end #function overview for MassVector

function mass_vector3d(m::Mesh, nz::Int)::MassVector3d
    # Creates a vector field on primal cell centers with UNDEF values
    F = Array{Float64}(undef, 3, m.nx, nz)
    return MassVector3d(m.nx, nz, F)
end # function mass_vector

function fill_field(m::Mesh, vector::MassVector3d, f::Function)
    # Fill MassVector with pointwise evaluation of function f at cell generators.
    # f is expected to return u and v from lat, lon in radians.
    for ix in 1:m.nx
        valu, valv = f(m.lat_i[ix], m.lon_i[ix])
        for iz in 1:vector.nz
            vector.F[:,ix,iz] = valu * m.ulam_i[:,ix] +
                                valv * m.uphi_i[:,ix]
        end
    end
    return
end

function meshconvert(m::Mesh, vec3d::MassVector3d, vec::MassVector)
    for ix in 1:m.nx
        for iz in 1:vec3d.nz
            vec.u[ix, iz] = dot(vec3d.F[:, ix, iz], m.ulam_i[:,ix])
            vec.v[ix, iz] = dot(vec3d.F[:, ix, iz], m.uphi_i[:,ix])
        end
    end
end

# MassVector3d structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# EdgeVector structure and relative stuff

struct EdgeVector
    nedge::Int
    nz::Int
    u::Array{Float64}
    v::Array{Float64}
end # struct EdgeVector

function overview(m::EdgeVector)::String
    # Print some properties of an edgevector (e.g. unit vector at edge centers)
    s = ""
    s = s * "This is an edge vector field with nedge, nz: " *
            string(m.nedge) * " " *
        string(m.nz) * "\n"
    minval, index = findmin(m.u)
    maxval, index = findmax(m.u)
    s = s * "min - max of the zonal component: " *
            string(minval) * " - " *
        string(maxval) * "\n"
    minval, index = findmin(m.v)
    maxval, index = findmax(m.v)
    s = s * "min - max of the meridional component: " *
            string(minval) * " - " *
        string(maxval) * "\n"
   
    return s
    
end #function overview for EdgeVectors

function edge_vector(m::Mesh, nz::Int)::EdgeVector
    # Creates a vector field on edge centers with UNDEF values
    u = Array{Float64}(undef, m.nedge, nz)
    v = Array{Float64}(undef, m.nedge, nz)
    return EdgeVector(m.nedge, nz, u, v)
end # function edge_vector

function fill_field(m::Mesh, vector::EdgeVector, f::Function)
    # Fill EdgeVector with pointwise evaluation of function f at cell generators.
    # f is expected to return u and v from lat, lon in radians.
    for ix in 1:m.nedge
        valu, valv = f(m.lat_e[ix], m.lon_e[ix])
        for iz in 1:vector.nz
            vector.u[ix,iz] = valu
            vector.v[ix,iz] = valv
        end
    end
    return
end
# EdgeVector structure and relative stuff
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Gradient calculation on a mesh.
# 
# gradient uses gradient_weights to return the gradient of a MassField
#    
# primal2dual uses barycentric weights to interpolate a scalar from the primal celle generators
#     onto the dual cell generators.


function gradient(m::Mesh, phic:: MassField, grad::MassVector3d)
    # Calculate the gradient of a MassField and store the result in MassVector grads
    # This uses precalculated weights.
    # This routine is NOT part of the initialization. It is optimized
    # (no trigonometric calculations)
    for iz in 1:grad.nz
        for ix in 1:m.nx
            grad.F[:, ix, iz] = sum( ( phic.F[m.primal_neighbour[ind, ix],iz]
	                               .- phic.F[ix,iz]
				     )
	                             * m.primal_grad3d[ind, ix, :]
				     for ind in 1:m.primal_deg[ix]
				   )
        end
    end
end

function gradient_limiter!(m::Mesh, phic:: MassField, grad::MassVector3d)
    # Limitates the gradient
    # This routine is NOT part of the initialization. It is optimized
    # (no trigonometric calculations)
    for ix in 1:m.nx
        for iz in grad.nz
            # calculate the min and max of phic over the current primal cell and its neighbour
            mini = phic.F[ix, iz]
            maxi = phic.F[ix, iz]
            phicenter = phic.F[ix, iz]
            for iedge in 1:m.primal_deg[ix]
                nei = m.primal_nei[iedge, ix]
                mini = min(mini, phic.F[nei, iz])
                maxi = max(maxi, phic.F[nei, iz])
            end
            # calculate the alpha multiplicator (alpha <=1)
            alpha = 1.0
            for iedge in 1:m.primal_deg[ix]
                edge_est = phicenter + dot(grad.F[:,ix,iz], m.cen2edge[:, iedge, ix])
                if edge_est > maxi
                    alpha = min(alpha, (maxi - phicenter) / (edge_est - phicenter))
                elseif edge_est < mini
                    alpha = min(alpha, (mini - phicenter) / (edge_est - phicenter))
                end                     
            end
            # apply the alpha mutliplicator
            grad.F[:,ix,iz] = alpha * grad.F[:,ix,iz]
        end 
    end
end

function primal2dual(m::Mesh, f::MassField, d::DualField)
    # Interpolate from primal to dual grid generators using barycentric weights.
    # This routine is NOT part of the initialization. It is optimized
    # (no trigonometric calculations)
    for iz in 1:f.nz
        for ix in 1:m.ndual
        d.F[ix,iz] = 0. 
        for iedge in 1:m.max_dual_deg
            wei = 1. - 2. * m.Riv2[iedge, ix]
            d.F[ix, iz] =  d.F[ix, iz] + wei * f.F[m.dual_vertex[iedge, ix], iz]
        end
    end # loop on ix
    end # loop on iz
    return
end

# End gradient calculation
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Speed reconstruction on a mesh


function normal2tangential(m::Mesh, v::NormalVector, tang::TangentialVector)
    # Reconstructs the tangential component  from the normal component for a vector known at edge centers
    # This routine is NOT part of the initialization. It is optimized
    # (no trigonometric calculations)

    for iedge in 1:v.nedge
        for iz in 1:v.nz
            tang.F[iedge, iz] = sum( v.F[m.edge_kite[ind, iedge],iz] * m.edge_perp[ind, iedge]
	                          for ind in 1:4  )
        end # loop on iz
    end # loop in iedge
    return
end

function speedreconst(m::Mesh, v::NormalVector, tang::TangentialVector, rec::EdgeVector)
    #   Constructs u and v from the normal and meridional component
    # This routine is NOT part of the initialization. It is optimized
    # (no trigonometric calculations)
    for iedge in 1:v.nedge
        for iz in 1:v.nz
            recvec = v.F[iedge, iz] * m.normalvec_edge[:,iedge] + tang.F[iedge, iz] * m.tangvec_edge[:,iedge]
            rec.u[iedge, iz] = dot(recvec, m.ulam_e[:,iedge])
            rec.v[iedge, iz] = dot(recvec, m.uphi_e[:,iedge])
        end # loop on iz
    end # loop in iedge
    return
end # function speedreconst

#End speed reconstruction
#---------------------------------------------------------------------------------------------

function create_netcdf_structure(m::Mesh, ncfile="GeoAdv_out.nc", nz = 1)
    
    FILLVALUE = -999.0
    
    ds = NCDataset(ncfile,"c")

    # Create global attributes
    ds.attrib["name"] = "GeoAdv_out"
    ds.attrib["description"] = "GeoAdv simulation output"
    ds.attrib["title"] = "GeoAdv simulation output"
    ds.attrib["Conventions"] = "CF-1.6"
    ds.attrib["model"] = "GeoAdv"
    
    # create dimensions
    defDim(ds,"lev", nz)
    defDim(ds,"cell", m.nx)
    defDim(ds,"nvertex", m.max_primal_deg)
    defDim(ds,"time_counter", Inf)
    
    # create geographic information
    
    # Centers
    lat = defVar(ds,"lat",Float64,("cell",))
    lat.attrib["standard_name"] = "latitude"
    lat.attrib["long_name"] = "Latitude"
    lat.attrib["units"] = "degrees_north"
    lat.attrib["bounds"] = "bounds_lat"
    lat[:] = m.lat_i * 180.0 / pi

    lon = defVar(ds,"lon",Float64,("cell",))
    lon.attrib["standard_name"] = "longitude"
    lon.attrib["long_name"] = "Longitude"
    lon.attrib["units"] = "degrees_east"
    lon.attrib["bounds"] = "bounds_lon"
    lon[:] = m.lon_i  * 180.0 / pi
    
    # Boundaries
    bounds_lat = defVar(ds,"bounds_lat",Float64,("nvertex", "cell"))
    bounds_lat.attrib["_FillValue"] = FILLVALUE
    bounds_lon = defVar(ds,"bounds_lon",Float64,("nvertex", "cell"))
    bounds_lon.attrib["_FillValue"] = FILLVALUE
    for ix in m.nx
        for iv = m.max_primal_deg
            if iv <= m.primal_deg[ix]
                bounds_lat[iv, ix] = m.primal_bounds_lat[iv, ix]
                bounds_lon[iv, ix] = m.primal_bounds_lon[iv, ix]
            else
                bounds_lat[iv, ix] = FILLVALUE
                bounds_lon[iv, ix] = FILLVALUE
            end
        end
    end
    
    # Time counter
    time_counter = defVar(ds, "time_counter", Float64, ("time_counter",))
    time_counter.attrib["axis"] = "T"
    time_counter.attrib["standard_name"] = "time"
    time_counter.attrib["long_name"] = "Time axis"
    time_counter.attrib["units"] = "non-dimensional"
#    close(ds)
    return ds
end

function create_mass_field(ds::NCDatasets.NCDataset, varname, long_name, units, coordinates)
    var = defVar(ds, varname, Float64, ("lev","cell","time_counter"))
    var.attrib["long_name"] = long_name
    var.attrib["units"] = units
    var.attrib["coordinates"] = coordinates 
    
    return nothing
end

function add_time_slot(ds::NCDatasets.NCDataset, t)
    slot = size(ds["time_counter"])[1]
    tc = ds["time_counter"]
    tc[slot+1:slot+1] = [ t ]    
    return nothing
end

function put_mass_field(ds::NCDatasets.NCDataset, varname, mf::MassField)
    slot = size(ds["time_counter"])[1]
    field = ds[varname]
    field[:,:,slot:slot] = mf.F
    return nothing
end

end