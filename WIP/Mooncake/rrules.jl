module CustomRules

# codual = reverse codual = primal + rdata
# fcodual = forward codual = primal + fdata

using Mooncake: Mooncake, CoDual, NoFData, NoRData, zero_fcodual, primal

using Main: f2!, f2!_custom

Mooncake.rrule!!(::CoDual{typeof(f2!_custom),NoFData}, fx) = f2!_rrule!!(fx)

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(f2!_custom), AbstractArray{<:AbstractFloat}}

function f2!_rrule!!(fx::CoDual{<:AbstractArray})
    # x==primal(fx) isa Array => fx::CoDual contain fdata fx.dx to be incremented in-place during the backward pass
    x, x_fdata, x_rdata = primal(fx), fx.dx, Mooncake.NoRData()
    # f2!(x) isnothing => NoFData(), NoRData()
    f2!_rdata = Mooncake.NoRData()
    # precompute gradient before f2! modifies x
    cosx = cos.(x)
    # forward pass
    y = zero_fcodual(f2!(x)) # ::CoDual{Nothing, NoFData}
    @info "rrule!!" typeof(x)

    # pullback closure, captures cosx and x_fdata
    function f2!_pullback!!(::Mooncake.NoRData)
        @. x_fdata *= cosx
        return f2!_rdata, x_rdata
    end
    return y, f2!_pullback!!
end

end

