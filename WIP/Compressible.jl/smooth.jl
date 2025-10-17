using Krylov: cg
using CFDomains.Stencils
using LoopManagers.ManagedLoops: @unroll

function smoothed(sph, Phis, Δ ; n=3, verbose=0)
    Ai = sph.Ai/mean(sph.Ai)
    h = Helmholtz(Δ^2/n, sph, similar(sph.le_de))
    Phis_smooth = Phis
    for _ in 1:n
        (Phis_smooth, _) = cg(h, Ai.*Phis_smooth ; verbose)
    end
    return Phis_smooth
end

struct Helmholtz{T, V<:VoronoiSphere{T}}
    Δ::T # smoothing length
    sph::V
    y::Vector{T} # temporary space
end
Base.size(h::Helmholtz) = length(h.sph.Ai), length(h.sph.Ai)
Base.eltype(::Helmholtz{T}) where T = T

function Base.:(*)(h::Helmholtz, x)
    Lx = zero(x)
    LinearAlgebra.mul!(Lx, h, x, false, true)
    return Lx
end

function normgrad(sph, x)
    g, y = similar(x), similar(sph.le_de)
    @inbounds for edge in eachindex(y)
        grad = Stencils.gradient(sph, edge)
        y[edge] = grad(x) # covariant gradient
    end
    @inbounds for (cell, deg) in enumerate(sph.primal_deg)
        @unroll deg in 5:7 begin
            dp = Stencils.dot_product(sph, cell, Val(deg))
            g[cell] = sqrt(dp(y,y))
        end
    end
    return g
end

function LinearAlgebra.mul!(Lx, h::Helmholtz, x, α, β)
    (; Δ, sph, y) = h
    (; primal_deg, le_de, Ai) = sph
    for edge in eachindex(y)
        grad = Stencils.gradient(sph, edge)
        y[edge] = le_de[edge]*grad(x) # contravariant gradient
    end
    mA = mean(Ai)
    for cell in eachindex(primal_deg)
        deg = primal_deg[cell]
        @unroll deg in 5:7 begin
            let dvg = Stencils.divergence(sph, cell, Val(deg))
                Lx[cell] = β*Lx[cell] + α*(Ai[cell]/mA)*(x[cell]-Δ*dvg(y))
            end
        end
    end
    return Lx
end
