"""
    import MemberFunctions: WithMembers, member_functions

    # declare that `MyType` has member functions
    struct MyType <: WithMembers end
    member_functions(::Type{<:MyType}) = (; hello, goodbye)

    # implement member functions as normal Julia functions
    # taking the object as first argument
    hello(::MyType) = println("Hello !")
    goodbye(::MyType) = println("Goodbye !")

    # test
    o = MyType()
    o.hello()
    o.goodbye()
    ? o.hello
"""
module MemberFunctions

abstract type WithMembers end
function member_functions end

Base.getproperty(o::WithMembers, prop::Symbol) = getprop(o, prop)
Base.propertynames(o::WithMembers) = propnames(o)

struct MemberCall{Fun, O}
    fun::Fun
    object::O
end
@inline (call::MemberCall)(args...) = call.fun(call.object, args...)

@inline member_names(x) = member_names(typeof(x))
@inline member_names(x::Type) = propertynames(member_functions(x))
@inline propnames(o) = (member_names(o)..., fieldnames(typeof(o))...)

@inline getprop(o, prop::Symbol) =
    (prop in fieldnames(typeof(o))) ? getfield(o,prop) : MemberCall(getproperty(member_functions(typeof(o)), prop), o)

@inline Base.show(io::IO, call::MemberCall) =
    println(io, "(args...)->$(call.fun)($(call.object), args...)")

# custom REPL help prompt

Base.Docs.getdoc(call::MemberCall) = Base.Docs.doc(call.fun)
Base.Docs.Binding(o::WithMembers, sym::Symbol) = property_binding(o, sym)

property_binding(object, sym) = member_binding(object, Val(sym), getproperty(object, sym), Val(sym in member_names(object)))
member_binding(_, _, call, is_a_member::Val{true}) = call.fun

function error_hinter(io, exc, argtypes, kwargs)
    val(::Type{Val{v}}) where v = v
    if exc.f == member_binding
        Object, field = argtypes[1], val(argtypes[2])
        print(io,
        "\nIt seems you asked for documentation on `object.$(field)` where `object::$Object`. ",
        "Since `$field` is a field of `object`, it fails. It would work if `$field` where ",
        "among its member functions, which are $(member_names(Object)).")
    end
end

function __init__()
    Base.Experimental.register_error_hint(error_hinter, MethodError)
end

end
