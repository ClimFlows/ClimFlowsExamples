function LRSW!(dm, ducov, vsphere, m, ucov)
    Ops.Divergence(vsphere, minus)(dm, ucov)  # ∂m/∂t = -∇⋅u
    Ops.Gradient(vsphere, minus)(ducov, m)    # ∂u/∂t = -∇m
    Ops.TRiSK(vsphere, subfrom)(ducov, ucov)  # ∂u/∂t -= u⟂
end

function sum_op2(f, tmp, op, app!)
    app!(tmp, op, f)
    return norm(tmp)
end

function test_op(q, tmp, op)
    @info "forward" sum_op2(q, tmp, op, Ops.apply_internal!)
    @code_warntype sum_op2(q, tmp, op, Ops.apply_internal!)
    
    backend = DI.AutoMooncake(; config=nothing)
    prep = DI.prepare_gradient(sum_op2, backend, q, Const(tmp), Const(op), Const(Ops.apply!));
    grad = DI.gradient(sum_op2, prep, backend, q, Const(tmp), Const(op), Const(Ops.apply!));
    prep_internal = DI.prepare_gradient(sum_op2, backend, q, Const(tmp), Const(op), Const(Ops.apply_internal!));
    grad_internal = DI.gradient(sum_op2, prep_internal, backend, q, Const(tmp), Const(op), Const(Ops.apply_internal!));
    @assert grad ≈ grad_internal    

    let
        run() = sum_op2(q, tmp,op, Ops.apply!)
        display(@benchmark $run())
    end
    let
        run() = DI.gradient(sum_op2, prep, backend, q, Const(tmp), Const(op), Const(Ops.apply!))
        display(@benchmark $run())
        @profview for _ in 1:10
            DI.gradient(sum_op2, prep, backend, q, Const(tmp), Const(op), Const(Ops.apply!))
        end
    end
end
