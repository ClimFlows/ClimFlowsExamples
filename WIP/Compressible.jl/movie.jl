@time_imports begin
    import ClimFlowsPlots.SpectralSphere as Plots
    using CFDomains.VerticalInterpolation: interpolate!
    using CairoMakie: Makie, record, Observable
    using NetCDF
end

function movie(model, diags, tape, var ; filename=joinpath(@__DIR__, "movie.mp4"))
    @info "Preparing plots..."
    diag(state) = var(open(diags; model, state))
    lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
    lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
    diag_obs = Observable(diag(tape[1]))
    fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)
    record(fig, filename, eachindex(tape)) do i
        @info "$filename: frame $i"
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
    return transpose(interpolated[:, :, 2])
end

function save(tape, filename = "SpectralHPE.nc"; vars...)

    nt = length(tape)
    for i in 0:nt
        @info "Writing state $i"
        open(diags; model, state=tape[max(1,i)]) do session
            for (name, var) in vars

                function write(data::Matrix)
                    vname = string(name)
                    (nx, ny) = size(data)
                    if i==0
                        NetCDF.nccreate(filename, vname, "lon", nx, "lat", ny, "time", nt)
                    else
                        data = reshape(data, (nx,ny,1))
                        ncwrite(data, filename, vname; start = [1, 1, i])
                    end
                end

                function write(data::Array{<:Any,3})
                    vname = string(name)
                    (nx, ny, nz) = size(data)
                    if i == 0
                        NetCDF.nccreate(filename, vname, "lon", nx, "lat", ny, "lev_$vname", nz, "time", nt)
                    else
                        data = reshape(data, (nx,ny,nz,1))
                        ncwrite(data, filename, vname; start = [1, 1, 1, i])
                    end
                end

                write(reverse(var(session); dims = 2))
            end
        end
    end
    NetCDF.ncclose(filename)
end

transp(x::Matrix) = transpose(x)
transp(x) = permutedims(x, (2,1,3))
ugradPhi = session->transp(session.ugradPhi)
ugradps = session->transp(session.ugradps)
ugradp = session->transp(session.ugradp)
gradPhi = session->transp(session.gradPhi)
ulat = session->transp(-session.uv.ucolat)
dulat = session->transp(-session.duv.ucolat)
W = session->transp(session.Phi_dot)
ps = session->transp(session.surface_pressure)
Omega = session->transp(session.Omega)
geopotential = session->transp(session.geopotential)
# ddPhi = session->transp(session.vertical_velocities.dthickness)
# geopotential = session->transp(session.vertical_velocities.dthickness)
dpdt = session->transp(session.vertical_velocities.dp)
W = session->transp(session.Phi_dot)
pressure = session->transp(session.pressure)
dmass = session->transp(session.dmasses.air)
T850 = var_ref(diags->diags.temperature)
Omega850 = var_ref(diags->diags.Omega)
W850 = var_ref(diags->diags.Phi_dot)
V850 = var_ref(diags->-diags.uv.ucolat)
