using Pkg; Pkg.activate(@__DIR__)
using Revise
using Symbolics

includet("simplify.jl")
includet("operators.jl")
using .Operators

pdv(f, vars::Tuple) = [expand_derivatives(Differential(v)(f)) v in vars]
pdv(fs::Tuple, vars::Tuple) = [expand_derivatives(Differential(v)(f)) for f in fs, v in vars]

consistent(ops, (dh,du,dv,dw)) = ops.simplify(ops.tilde(du,dv,dw))

#=
    F  = hu
    q = (∇×u+f)/h
    hₜ = -∇F
    uₜ = - q × F - ∇ ((u⋅u).2+h)

h → H+h ⟹  hₜ = -∇(Hu)
            uₜ = - f×u - ∇h

u,h → U + u, H+h ⟹
    q = ∇×u/H - fh/H^2
    F = Hu + hU
    hₜ + ∇(Hu) = -∇(hU)
    uₜ + f×u + ∇ (h+U⋅u) = -q×HU - (f/H)×(hU)
=#

function lsw(ops, h, u, v, w)
    du, dv, dw = ops.grad(-h)
    up, vp, wp = ops.perp(u,v,w)
    dh = -ops.dvg(u,v,w)
    return dh, du+up, dv+vp, dw+wp
end

#===============================================================#

@syms a b c u v w h

ops = Operators.ops(a,b,c);
(dh,du,dv,dw) = model = lsw(ops, h, u, v, w)
Jac = pdv(model, (h,u,v,w))
za, zb = ops.curl(ops.grad(h)...)

@info "checks" consistent(ops, model) ops.symmetrize(Jac) ops.simplify(za) ops.simplify(zb)
