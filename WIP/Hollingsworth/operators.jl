module Operators

using Base: Fix1
using Symbolics: iscall, operation, arguments, degree, substitute, expand

struct Operator{Op, ABC}
    operator::Op
    abc::ABC
end
@inline (op::Operator)(args...) = op.operator(op.abc..., args...)

function ops(a,b,c)
    return map(op->Operator(op, (a,b,c)), (; dvg, curl, grad, perp, tilde, conj, symmetrize, simplify))
end

dvg(a, b, c, u, v, w) = (a^3-b^3*c^3)*u + (b^3-a^3*c^3)*v + (c^3-b^3*a^3)*w
curl(a, b, c, u, v, w) = (a*c^2*u + b*a^2*v + c*b^2*w), -(a*b^2*u + b*c^2*v + c*a^2*w)
grad(a, b, c, q) = (a^3-b^3*c^3)*q, (b^3-c^3*a^3)*q, (c^3-a^3*b^3)*q
perp(a,b,c,u,v,w) = perp_(a,b,c,v,w), perp_(b,c,a,w,u), perp_(c,a,b,u,v)
tilde(a, b, c, u, v, w) = tilde_(a,b,c)*u + tilde_(b,c,a)*v + tilde_(c,a,b)*w

tilde_(a, b, c) = a^3*(b^6 + c^6)/6 + (b^3*c^3 + a^3)/3

function perp_(a,b,c,v,w) 
    A,B,C = a^3, b^3, c^3
    AB, BC, CA = A*B, B*C, C*A
    return (B*BC+A*CA+2*(AB+C))*v - (AB*A+C*BC+2*(CA+B))*w
end

#=========================
   B*BC==B     AB==AB*A
 //       \\ //       \\
     BC     u     A
 \\       // \\       //
   C*BC==C     CA==CA*A
==========================#

symmetrize(a,b,c, J::Matrix) = [simplify(a,b,c,x) for x in J+transp(conj(a,b,c, J))]
transp(A::Matrix) = permutedims(A, (2,1))
conj(a,b,c, expr) = substitute(expr, Dict(a=>b*c, b=>a*c, c=>a*b))

function simplify(a,b,c, expr)
    function eliminate_abc(expr)
        n = min(degree(expr, a), degree(expr, b), degree(expr, c))
        return expand(expr/((a*b*c)^n))
    end

    expr = expand(expr)
    if is_sum(expr)
        terms = arguments(expr)
        simplified = map(eliminate_abc, terms)
        return sum(simplified)
    elseif is_prod(expr)
        return eliminate_abc(expr)
    else
        return expr
    end
end

is_sum(expr) = iscall(expr) && operation(expr) == +
is_prod(expr) = iscall(expr) && operation(expr) == *

end # module
