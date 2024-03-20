const debug=false
include("preamble.jl")

@time_imports using OrdinaryDiffEq, Zygote, SciMLSensitivity, AbbreviatedStackTraces, RecursiveArrayTools

function simple_ode(x, p, t)

#    @info typeof(x) typeof(p) typeof(_zero), typeof(_one)
    p.x[1] + p.x[2]+ x
end
# simple_ode(x, p, t) = x
simple_ode(x) = x

function setup_debug(; alg=OrdinaryDiffEq.RK4(), dt=0.1, model=simple_ode) # , options...)
    tspan = (0, 10*dt)

    function forward(spec0)
        p = ArrayPartition(zero.(spec0), one.(spec0))
        problem = OrdinaryDiffEq.ODEProblem(model, spec0, tspan, p)
        sol = OrdinaryDiffEq.solve(problem ; alg, dt, adaptive=false, saveat=last(tspan), options...)
        return sol.u[2]
    end
    return model, randn(100), forward
end

struct Loss{Scheme,Target}
    scheme::Scheme
    target::Target
end

function (loss::Loss)(state)
    (; scheme, target) = loss
    new = scheme(state)
    return sum(abs2, new - target)
end

options = (; sensealg=SciMLSensitivity.InterpolatingAdjoint(; autojacvec=SciMLSensitivity.ZygoteVJP()))

model, state, forward = setup_debug()
dstate = randn(length(state)) .* state;
target = forward(forward(state))

loss1(state) = sum(abs2, state)
loss2 = Loss(model, dstate)
loss3 = Loss(forward, target)

Zygote.gradient(loss1, state)
Zygote.gradient(loss2, state)
Zygote.gradient(loss3, state)
