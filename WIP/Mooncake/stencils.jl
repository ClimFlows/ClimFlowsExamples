using ManagedLoops: @unroll

# each operator holds the data needed to compute itself and also to compute its adjoint
abstract type DiffStencil{In,Out} end

ignore(x,y) = y # default action

struct Gradient{Action} <: DiffStencil{1,1}
    action::Action # how to combine op(input) with output
    edge_left_right::Matrix{Int32}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{Int32}
end
Gradient(sph) = Gradient(ignore, sph)
Gradient(action, sph) = Gradient(action, sph.edge_left_right, sph.primal_deg, sph.primal_edge, sph.primal_ne)

# divergence operator, without dividing by Ai : result is a 2-form
struct DivForm{Action} <: DiffStencil{1,1}
    action::Action # how to combine op(input) with output
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{Int32}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end
DivForm(action, sph) = DivForm(action, sph.primal_deg, sph.primal_edge, sph.primal_ne, sph.edge_left_right)

struct TRiSK{Action, F} <: DiffStencil{1,1}
    action::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end
TRiSK(sph) = TRiSK(ignore, sph)
TRiSK(action, sph) = TRiSK(action, sph.trisk_deg, sph.trisk, sph.wee)

apply!(output, stencil::DiffStencil{1,1}, input) = apply!_internal(output, stencil, input)

@inline function apply!_internal(output, op::Gradient, input)
    (; action) = op
    @inbounds for edge in eachindex(output)
        grad = Stencils.gradient(op, edge)
        output[edge] = action(output[edge], grad(input))
    end
    return nothing
end

@inline function apply!_internal(output, op::DivForm, input)
    (; action) = op
    @inbounds for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            dvg = Stencils.div_form(op, cell, Val(deg))
            output[cell] = action(output[cell], dvg(input))
        end
    end
    return nothing
end

@inline function apply!_internal(output, op::TRiSK, input)
    (; action) = op
    for edge in eachindex(output)
        deg = op.trisk_deg[edge]
        @unroll deg in 9:11 begin
            trsk = Stencils.TRiSK(op, edge, Val(deg))
            output[edge] = action(output[edge], trsk(input))
        end
    end
    return nothing
end

module StencilRules

# codual = reverse codual = primal + rdata
# fcodual = forward codual = primal + fdata

using Mooncake: Mooncake, CoDual, NoTangent, NoFData, NoRData, zero_fcodual, primal
using Main: apply!, apply!_internal, ignore, DiffStencil, Gradient, DivForm, TRiSK

Mooncake.tangent_type(::Type{<:DiffStencil}) = NoTangent

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoNumber{F} = CoDual{F,NoFData}
const CoStencil{A,B} = CoDual{<:DiffStencil{A,B}, NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(apply!), Vararg}
Mooncake.rrule!!(::CoFunction(apply!), fx::Vararg) = apply!_rrule!!(fx...)

function apply!_rrule!!(foutput::CoVector{F}, op::CoStencil{1,1}, finput::CoVector{F}) where F
    output, stencil, input = primal(foutput), primal(op), primal(finput)
    output_ = copy(output) # to undo mutation during backward pass    
    dout, din, adj = tangent(foutput), tangent(finput), adjoint(stencil)
    function apply!_pullback!!(::NoRData)
        @. output = output_ # undo mutation
        apply!_internal(din, adj, dout)
        return NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
    end
    apply!_internal(output, stencil, input)
    return zero_fcodual(nothing), apply!_pullback!!
end

adjoint(op::Gradient{typeof(ignore)}) = DivForm(subfrom, op)
adjoint(op::TRiSK{typeof(ignore)}) = TRiSK(subfrom, op)
subfrom(x,y) = x-y

end
