cd(@__DIR__)
import Pkg;
Pkg.activate(@__DIR__); #src
unique!(push!(LOAD_PATH, "$(@__DIR__)/../include/"))

using InteractiveUtils

const start_time = time()
toc(str) = "At t=$(time()-start_time)s : $str"

try_import(mod) = (mod in keys(Pkg.project().dependencies)) && @eval import $(Symbol(mod))

@time_imports begin
    try_import("SIMDFunctions")
    try_import("KernelAbstractions")
end

try
    import LoopManagers
    ## these functions return a computing backend
    ## defining functions rather than const variables ensures that we re-initialize backends each time we re-run
    plain(args...) = LoopManagers.PlainCPU()
    omp(args...) = LoopManagers.MainThread(plain())
    threads(args...) = LoopManagers.MultiThread(plain())
    SIMD(args...) = LoopManagers.VectorizedCPU(args...)
    ompSIMD(args...) = LoopManagers.MainThread(SIMD(args...))
    autotune() = LoopManagers.tune()
    tSIMD(args...) = LoopManagers.MultiThread(SIMD(args...))
    fake(args...) = LoopManagers.FakeGPU(tSIMD(args...))
catch
end

if (debug) #src
    @eval macro fast(code) #src
        return esc(code) #src
    end #src
else #src
    @eval macro fast(code)
        return esc(quote
            @inbounds $code
        end)
    end
end #src

# orthographic plot; requires `using GeoMakie`

@views function upscale(field)
    field = 0.5*(field[1:2:end, :] + field[2:2:end,:])
    field = 0.5*(field[:, 1:2:end] + field[:,2:2:end])
end

function orthographic(lons, lats, field; options...)
    fig = Figure()
    ga = GeoAxis(fig[1, 1]; dest = "+proj=ortho +lon_0=19 +lat_0=50")
    surface!(ga, lons, lats, field; shading = NoShading, options...)
    lines!(ga, GeoMakie.coastlines())
    return fig
end

# convenience functions
bounds_lon(lons) = bounds(lons[end]-360, lons, lons[1]+360)
bounds_lat(lats) = bounds(-90.0, lats, 90.0)
function bounds(x0, x, xend)
    x = (x[2:end] + x[1:end-1])/2
    return [x0, x..., xend]
end
