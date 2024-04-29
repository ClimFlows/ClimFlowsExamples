@time_imports begin
    using NetCDF: ncread
    using LazyArtifacts
    import GeometryBasics as GB
    import GFSphericalInterpolation as SI
    import Makie
    import CairoMakie
    using ColorSchemes
#    import GFDomains
end

function read_mesh(meshname; Float = Float32)
    meshname = Base.Filesystem.joinpath(artifact"VoronoiMeshes", "VoronoiMeshes", meshname)
    function readvar(varname)
        ncread(meshname, varname)
    end

#    @info toc("Opening $meshname")
    mesh = CFDomains.VoronoiSphere(; prec = Float) do name
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
    mesh = CFDomains.VoronoiSphere(; prec = Float) do name
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

function plot_2D(sphere, diags, diagnose)
    lons, lats = -180:1:180, -90:1:90
    interp = SI.lonlat_interp(sphere, lons, lats)
    get(diags) =
        let (color, title) = diagnose(diags)
            transpose(interp(color)), title
        end
    color, title = get(diags)

    ## levels = floor(minimum(color)):0.5:ceil(maximum(color))
    color = Makie.Observable(color)
    fig = Figure(resolution = (720, 360))
    ax = Axis(fig[1, 1], aspect = 2)
    Makie.heatmap!(ax, color; colormap = :hot)
    ## Makie.contourf!(ax, lons, lats, color ; colormap = :hot )
    return (; fig, get, color)
end

function run_movie_2D(sphere, diags, diagnose, loop, moviename)
    fig, get, color = plot_2D(sphere, diags, diagnose)
    record(
        fig,
        moviename,
        loop;
        framerate = 20,
        compression = 0,
        profile = "high444",
    ) do (iter, time)
        GFRegistries.reset!(diags)
        color[], title = get(diags)
        if mod(iter, 10) == 0
            hour = "$(time/3600)h"
            @info toc("Wrote frames for t<=$hour")
        end
    end
    return
end

function run_movie_3D(sphere, diags, diagnose, loop, moviename, zoom = 1.6)
    makiemesh = let (lon, lat, vertex) = (sphere.lon_i, sphere.lat_i, sphere.dual_vertex)
        xyz(lon, lat) = cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)
        nodes = [GB.Point3f(xyz(lon[i], lat[i])) for i in eachindex(lon)]
        faces = [
            GB.GLTriangleFace((vertex[1, i], vertex[2, i], vertex[3, i])) for
            i in axes(vertex, 2)
        ]
        GB.Mesh(nodes, faces)
    end
    color, title = diagnose(diags)
    color = Makie.Observable(color)
    fig, ax, obj = Makie.mesh(makiemesh, color = color)
    Makie.scale!(ax.scene, zoom, zoom, zoom)
    record(fig, moviename, loop; framerate = 20) do (iter, time)
        GFRegistries.reset!(diags)
        color[], title = diagnose(diags)
        if mod(iter, 10) == 0
            hour = "$(time/3600)h"
            @info toc("Wrote frames for t<=$hour")
        end
    end
    return
end

function plot_voronoi_3D(sphere, color, title, zoom = 1.6)
    makiemesh = let (lon, lat, vertex) = (sphere.lon_i, sphere.lat_i, sphere.dual_vertex)
        xyz(lon, lat) = cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)
        nodes = [GB.Point3f(xyz(lon[i], lat[i])) for i in eachindex(lon)]
        faces = [
            GB.GLTriangleFace((vertex[1, i], vertex[2, i], vertex[3, i])) for
            i in axes(vertex, 2)
        ]
        GB.Mesh(nodes, faces)
    end
    color = Makie.Observable(color)
    fig, ax, obj = Makie.mesh(makiemesh, color = color)
    Makie.scale!(ax.scene, zoom, zoom, zoom)
    return fig, color
end
