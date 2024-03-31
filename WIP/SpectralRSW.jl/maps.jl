# ortographic
function orthographic(lons, lats, field; options...)
    fig = Figure()
    ga = GeoAxis(fig[1, 1]; dest = "+proj=ortho +lon_0=19 +lat_0=50")
    surface!(ga, lons, lats, field; shading = NoShading, options...)
    lines!(ga, GeoMakie.coastlines())
    return fig
end

bounds_lon(lons) = bounds(lons[end]-360, lons, lons[1]+360)
bounds_lat(lats) = bounds(-90.0, lats, 90.0)
function bounds(x0, x, xend)
    x = (x[2:end] + x[1:end-1])/2
    return [x0, x..., xend]
end

#=
# First, make a surface plot
lons = -180:180
lats = -90:90
field = [exp(cosd(l)) + 3(y/90) for l in lons, y in lats]

fig = Figure()
ax = GeoAxis(fig[1,1])
sf = surface!(ax, lons, lats, field; shading = NoShading)
cb1 = Colorbar(fig[1,2], sf; label = "field", height = Relative(0.65))

using GeoMakie.GeoJSON
countries_file = GeoMakie.assetpath("vector", "countries.geo.json")
countries = GeoJSON.read(read(countries_file, String))

n = length(countries)
hm = poly!(ax, countries; color= 1:n, colormap = :dense,
    strokecolor = :black, strokewidth = 0.5,
)
translate!(hm, 0, 0, 100) # move above surface plot

fig
=#
