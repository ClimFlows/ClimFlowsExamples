@time_imports begin
    import GeometryBasics as GB
    import GFSphericalInterpolation as SI
    import Makie
    import CairoMakie
    using ColorSchemes
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

function plot_voronoi_3D(sphere, color, title ; zoom = 1.6)
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
