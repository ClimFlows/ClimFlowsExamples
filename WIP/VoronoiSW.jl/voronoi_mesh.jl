@time_imports begin
    using NetCDF: ncread
    using LazyArtifacts
    import GeometryBasics as GB
    import GFSphericalInterpolation as SI
#    import GFDomains
end

function read_mesh(meshname; Float = Float32)
    meshname = Base.Filesystem.joinpath(artifact"VoronoiMeshes", "VoronoiMeshes", meshname)
    function readvar(varname)
        ncread(meshname, varname)
    end

#    @info toc("Opening $meshname")
    mesh = GFDomains.VoronoiSphere(; prec = Float) do name
        if name == :primal_num
            return length(readvar("primal_deg"))
        elseif name == :dual_num
            return length(readvar("dual_deg"))
        elseif name == :edge_num
            return length(readvar("trisk_deg"))
        elseif name == :le_de
            ## the DYNAMICO mesh file contains 'le' and 'de' separately
            le = readvar("le")
            de = readvar("de")
            return le ./ de
        elseif name == :primal_perot_cov
            return perot_cov!(
                (
                    readvar(name) for
                    name in ("primal_perot", "le", "de", "primal_deg", "primal_edge")
                )...,
            )
        else
            return readvar(String(name))
        end
    end

    @info """
Read $Float mesh from $meshname :
    Mesh has $(length(mesh.Ai)) cells, $(length(mesh.Av)) dual cells, $(length(mesh.de)) edges.
    Cells have $(minimum(mesh.primal_deg))-$(maximum(mesh.primal_deg)) edges.
    Edges have $(minimum(mesh.trisk_deg))-$(maximum(mesh.trisk_deg)) TRiSK neighbours."""

    return mesh
end

const AA{N,T} = AbstractArray{T,N}
slice(rank, var::AA{1}) = var[rank+1]
slice(rank, var::AA{2}) = var[:,rank+1]
slice(rank, var::AA{3}) = var[:,:,rank+1]
slice(rank, var::AA{4}) = var[:,:,:,rank+1]

function read_pmesh(ncvars, rank; Float = Float32)
    readvar(varname) = slice(rank, ncvars[varname])
    merge(x,y) = vcat(x',y')

    @info toc("Rank $rank opening mesh file")
    mesh = GFDomains.VoronoiSphere(; prec = Float) do name
        if name == :le_de
            ## the DYNAMICO mesh file contains 'le' and 'de' separately
            data = readvar("le")./readvar("de")
        elseif name == :edge_left_right
            data = merge(readvar("left"), readvar("right"))
        elseif name == :edge_down_up
            data = merge(readvar("down"), readvar("up"))
        elseif name == :primal_perot_cov
            varnames = "primal_perot", "le", "de", "primal_deg", "primal_edge"
            data = perot_cov!(map(readvar, varnames)...)
        else
            data = readvar(String(name))
        end
        return data
    end

    @info toc(
        """
Rank $rank read $Float mesh from $meshname :
    Mesh has $(mesh.primal_num) cells, $(mesh.dual_num) dual cells, $(mesh.edge_num) edges.
    Cells have $(minimum(mesh.primal_deg))-$(maximum(mesh.primal_deg)) edges.
    Edges have $(minimum(mesh.trisk_deg))-$(maximum(mesh.trisk_deg)) TRiSK neighbours.""",
    )

    return mesh
end

function perot_cov!(perot, le, de, degree, edge)
    ## the Perot weights found in the DYNAMICO mesh file
    ## assume *contravariant* components as inputs
    ## we want to apply them to covariant data (momentum)
    ## => multiply weights by le/de
    for ij in eachindex(degree)
        deg = degree[ij]
        for e = 1:deg
            le_de = le[edge[e, ij]] * inv(de[edge[e, ij]])
            perot[e, ij, 1] *= le_de
            perot[e, ij, 2] *= le_de
            perot[e, ij, 3] *= le_de
        end
    end
    return perot
end
