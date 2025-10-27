function f2!(x)
    @. x = sin(x)
    return nothing
end
f2!_custom(x) = f2!(x)

function loss(x, f!) 
    s = sum(x)
    for _ in 1:3
        f!(x)
        s += sum(x)
    end
    return s
end

megabytes(x) = div(Base.summarysize(x), 1024*1024)
function grad_loss(x, f!)
    backend = AutoMooncake(; config=nothing)
    prep = prepare_gradient(loss, backend, copy(x), DI.Constant(f!));
    @info "grad_loss" typeof(prep) megabytes(prep)
    # display(@benchmark gradient(loss, $prep, $backend, $(copy(x)), $(DI.Constant(f!))))
    return gradient(loss, prep, backend, copy(x), DI.Constant(f!))
end

x = randn(10,10);
@showtime g_custom = grad_loss(x, f2!_custom);
@showtime g = grad_loss(x, f2!);
@info "check" g ≈ g_custom

fixed(x) = FixedSizeArray(copy(x))
gradient(sum, AutoMooncake(), copy(x))
gradient(sum, AutoMooncake(), fixed(x))
gradient(loss, AutoMooncake(), fixed(x), DI.Constant(f2!))
gradient(loss, AutoMooncake(), fixed(x), DI.Constant(f2!_custom))
