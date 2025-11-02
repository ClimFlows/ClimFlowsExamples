module CustomRules

using Mooncake: Mooncake, CoDual, NoFData, NoRData, zero_fcodual, primal, tangent

using Main: f2!, f2!_custom

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(f2!_custom), AbstractArray{<:AbstractFloat}}

CoFunction(f) = CoDual{typeof(f), NoFData}
function Mooncake.rrule!!(::CoFunction(f2!_custom), fx::CoDual{<:AbstractArray})
    x, dx = primal(fx), tangent(fx)
    xx = copy(x) # copy x before f2! modifies it
    function my_pullback!!(::NoRData) # captures x, xx, dx
        @. dx *= cos(xx)
        @. x = xx # undo mutation
        return NoRData(), NoRData()
    end
    return zero_fcodual(f2!(x)), my_pullback!!
end

end
