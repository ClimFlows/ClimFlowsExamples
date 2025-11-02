module MooncakeFixedSizeArrays

using FixedSizeArrays: FixedSizeArray as FSArray
using Mooncake: CoDual, NoFData, NoRData, NoPullback, MaybeCache, SetToZeroCache,
        zero_fcodual, primal, lgetfield
import Mooncake

Mooncake.tangent_type(::Type{F}) where {F<:FSArray} = F
Mooncake.fdata_type(::Type{F}) where {F<:FSArray} = F
Mooncake.rdata_type(::Type{F}) where {F<:FSArray} = NoRData
Mooncake.tangent(f::FSArray, ::NoRData) = f

Mooncake.zero_tangent_internal(x::FSArray, ::MaybeCache) = zero(x)

function Mooncake.set_to_zero_internal!!(c::SetToZeroCache, x::FSArray)
    Mooncake.set_to_zero_internal!!(c, x.mem)
    return x
end

const CoVal{sym} = CoDual{Val{sym}}
@inline Mooncake.rrule!!(f::CoDual{typeof(lgetfield)}, rs::CoDual{<:FSArray}, ri::CoVal) = getprop(rs, ri), NoPullback(f, rs, ri)
@inline getprop(rs, ::CoVal{:size}) = zero_fcodual(rs.x.size)
@inline getprop(rs, ::CoVal{:mem}) = CoDual(rs.x.mem, rs.dx.mem)

end
