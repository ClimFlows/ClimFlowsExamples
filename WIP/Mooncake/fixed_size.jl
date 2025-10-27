module MooncakeFixedSide

using FixedSizeArrays: FixedSizeArray
using Mooncake: CoDual, NoFData, NoRData, MaybeCache, SetToZeroCache,
        zero_fcodual, primal, lgetfield
import Mooncake

Mooncake.tangent_type(::Type{F}) where {F<:FixedSizeArray} = F
Mooncake.fdata_type(::Type{F}) where {F<:FixedSizeArray} = F
Mooncake.rdata_type(::Type{F}) where {F<:FixedSizeArray} = NoRData
Mooncake.tangent(f::FixedSizeArray, ::NoRData) = f

Mooncake.zero_tangent_internal(x::FixedSizeArray, ::MaybeCache) = zero(x)
function Mooncake.set_to_zero_internal!!(c::SetToZeroCache, x::FixedSizeArray)
    Mooncake.set_to_zero_internal!!(c, x.mem)
    return x
end

const CoVal{sym} = CoDual{Val{sym}}
@inline Mooncake.rrule!!(f::CoDual{typeof(lgetfield), NoFData}, rs::CoDual{<:FixedSizeArray}, ri::CoVal{:length}) = lgetfield_rrule!!(f, rs, ri)
@inline Mooncake.rrule!!(f::CoDual{typeof(lgetfield), NoFData}, rs::CoDual{<:FixedSizeArray}, ri::CoVal{:size}) = lgetfield_rrule!!(f, rs, ri)
@inline Mooncake.rrule!!(f::CoDual{typeof(lgetfield), NoFData}, rs::CoDual{<:FixedSizeArray}, ri::CoVal{:mem}) = lgetfield_rrule!!(f, rs, ri)

@inline function lgetfield_rrule!!(f, rs, ri::CoDual{<:Val{field}}) where field
    pb!! = Mooncake.NoPullback(f, rs, ri)
    return zero_fcodual(getfield(primal(rs), field)), pb!!
end

end
