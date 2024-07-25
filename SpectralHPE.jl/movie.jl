@time_imports begin
    import ClimFlowsPlots.SpectralSphere as Plots
    using CairoMakie: Makie, record
end

@info "Preparing plots..."

lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)

diag_obs = Makie.Observable(diag(state0))
# main thread making the movie
record(fig, "$(@__DIR__)/T850.mp4", 1:N) do i
    @info "t=$(div(interval*i,3600))h"
    diag_obs[] = take!(channel)
    if mod(params.interval * i, 86400) == 0
        @info "day $(i/4)"
        display(heatmap(transpose(diag_obs[])))
    end
end
