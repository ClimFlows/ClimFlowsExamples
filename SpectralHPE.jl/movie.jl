@time_imports begin
    import ClimFlowsPlots.SpectralSphere as Plots
    using CFDomains.VerticalInterpolation: interpolate!
    using CairoMakie: Makie, record
    using NetCDF
end

function movie(model, diags, tape, var ; filename=joinpath(@__DIR__, "movie.mp4"))
    @info "Preparing plots..."
    diag(state) = var(open(diags; model, state))
    lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
    lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
    diag_obs = Makie.Observable(diag(tape[1]))
    fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)
    record(fig, filename, eachindex(tape)) do i
        @info "frame $i"
        diag_obs[] = diag(tape[i])
    end
end

var_ref(var) = function(session)
    model = session.model
    interpolated = interpolate!(
        model.mgr,
        void,
        model.domain,
        var(session),
        session.pressure,
        [85000.0, 50000.0],
        false,
    )
    return transpose(interpolated[:, :, 1])
end

function save(tape, filename = "data.nc"; vars...)
    nt = length(tape)
    for i in 1:nt
        open(diags; model, state=tape[i]) do session
            for (name, var) in vars
                vname = string(name)
                data = reverse(var(session); dims = 2)
                (nx, ny) = size(data)
                i == 1 && NetCDF.nccreate(filename, vname, "lon", nx, "lat", ny, "time", nt)
                data = reshape(data, (nx,ny,1))
                ncwrite(data, filename, vname; start = [1, 1, i])
            end
        end
    end
end
