function f2!(x)
    @. x = sin(x)
    return nothing
end
f2!_custom(x) = f2!(x)

function loss(x, f!) 
    s = sum(x)
    for _ in 1:2
        f!(x)
        s += sum(x)
    end
    return s
end

megabytes(x) = div(Base.summarysize(x), 1024*1024)
function grad_loss(x, f!)
    @info "grad_loss" f! typeof(x)
    backend = DI.AutoMooncake()
    prep = DI.prepare_gradient(loss, backend, x, DI.Constant(f!));
    display(@benchmark DI.gradient(loss, $prep, $backend, $(copy(x)), $(DI.Constant(f!))))
    return DI.gradient(loss, prep, backend, copy(x), DI.Constant(f!))
end

fixed(x) = FixedSizeArray(copy(x))

x = randn(10,10);
g_manual = @. 1+cos(x)+cos(x)*cos(sin(x)) # (d/dx)(x+sin(x)+sin(sin(x))

@info "=================================================================" 
g = grad_loss(copy(x), f2!)
@info "check" g ≈ g_manual

@info "=================================================================" 
g_custom = grad_loss(copy(x), f2!_custom)
@info "check" g_custom ≈ g_manual

@info "=================================================================" 
g_fixed = grad_loss(fixed(x), f2!)
@info "check" g_fixed ≈ g_manual

@info "=================================================================" 
g_custom_fixed = grad_loss(fixed(x), f2!_custom)
@info "check" g_custom_fixed ≈ g_manual

g_custom = DI.gradient(loss, DI.AutoMooncake(), copy(x), DI.Constant(f2!_custom))
@info "check" g_custom ≈ g_manual
