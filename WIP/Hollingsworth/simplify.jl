#=
function simplify_monomial(expr)
    @variables a b c
    n = minimum(v->Symbolics.degree(expr, v), (a, b, c))
    return expand(expr/((a*b*c)^n))
end

function simplexpand(expr)
    expr = expand(expr)
    if iscall(expr) && operation(expr) == +
        terms = Symbolics.arguments(expr)
        simplified = map(simplify_monomial, terms)
        return sum(simplified)
    else
        return expr
    end
end
=#

