using Pkg
Pkg.activate(joinpath(@__DIR__,"Plots"))

using InteractiveUtils
@time_imports begin
    using ClimFlowsPlots: SphericalInterpolations as SI
    using GeoMakie, CairoMakie, ClimFlowsPlots

    using Serialization
    using Logging: global_logger
    using TerminalLoggers: TerminalLogger
    using ProgressLogging

    using CFTimeSchemes, LoopManagers, CFCompressible
end

function get_T10(diags, model, _, state)
    session = open(diags; model, state)
    return session.temperature_i[10,:]
end

function get_Teq(diags, model, to_lonlat, state)
    session = open(diags; model, to_lonlat, state)
    T = session.temperature
    return permutedims(T[1:end-5, div(1+size(T,2),2), :], (2,1))
end

function movie(field, get, tape, filename)
    @withprogress name="Recording movie" record(fig, "$(@__DIR__)/T10.mp4", eachindex(tape)) do i
        @logprogress i / length(tape)
        field[] = get(diags, model, interp, tape[i])
    end
end

global_logger(TerminalLogger())

@info "Loading data..."

data = deserialize(joinpath(@__DIR__, "output_compressible_voronoi.jld"))

(; scheme_VFCE, tape_FCE) = data
model = scheme_VFCE.model
sphere = model.domain.layer
@. sphere.lon_i += 3pi/2
@. sphere.lon_e += 3pi/2
@. sphere.lon_v += 3pi/2
lons = collect(-180:180)
lats = collect(-90:90)
tree = SI.spherical_tree(sphere)
interp = SI.lonlat_interp(lons, lats, sphere, tree)
diags = CFCompressible.Diagnostics.diagnostics()

let
    @info "Preparing 2D plot..."
    Teq = get_Teq(diags, model, interp, tape_FCE[end])
    Tmin, Tmax = extrema(Teq)
    Teq = GeoMakie.Makie.Observable(Teq)
    fig = Makie.Figure(size = (1440, 720))
    ax = Makie.Axis(fig[1, 1], aspect = 2)
    Makie.contourf!(ax, Teq ; levels=LinRange(Tmin, Tmax, 20))

    @withprogress name="Recording Teq movie" record(fig, "$(@__DIR__)/Teq.mp4", eachindex(tape_FCE)) do i
        @logprogress i / length(tape_FCE)
        Teq[] = get_Teq(diags, model, interp, tape_FCE[i])
    end
end

let
    @info "Preparing stereographic plot..."
    T10 = get_T10(diags, model, nothing, tape_FCE[end])
    colorrange = extrema(T10)
    T10 = GeoMakie.Makie.Observable(T10)
    fig = ClimFlowsPlots.VoronoiSphere.plot_orthographic(T10, model.domain.layer; colorrange);

    @withprogress name="Recording T10 movie" record(fig, "$(@__DIR__)/T10.mp4", eachindex(tape_FCE)) do i
        @logprogress i / length(tape_FCE)
        T10[] = get_T10(diags, model, nothing, tape_FCE[i])
    end
end
