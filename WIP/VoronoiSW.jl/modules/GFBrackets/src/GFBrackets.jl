module GFBrackets

using ManagedLoops: @loops, @unroll
const debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"]!="")

if(debug)
    @eval macro fast(code) return esc(code) end
else
    @eval macro fast(code) return esc(quote @inbounds $code end) end
end

include("voronoi.jl")

end # module
