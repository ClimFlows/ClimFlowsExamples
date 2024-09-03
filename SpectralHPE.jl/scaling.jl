# create N models for benchmarking
function duplicate_model(choices, params, model, nt)
    sph = SHTnsSphere(model.domain.layer, nt)
    mgr = MultiThread(VectorizedCPU(), nt)
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    return HPE(params, mgr, sph, model.vcoord, surface_geopotential, model.gas)
end

function benchmark(choices, params, sph, mgrs)
    for mgr in mgrs
        # NB : spherical harmonics are multithread in all cases !
        @info "===== Time needed to simulate 6h with $mgr ====="
        info, state0 = setup(choices, params, sph, mgr)
        scratch = scratch_remap(info.diags, info.model, state0)
        (; courant, interval) = params
        dt = max_time_step(info, courant, interval, state0)
        timeloop = TimeLoop(info, state0, 0.0, true) # zero time step for benchmarking
        run_benchmark(timeloop, state0, dt, interval; hours = 0) # compile
        @time run_benchmark(timeloop, state0, dt, interval, scratch)
        #        @profview run_benchmark(timeloop, state0, dt, interval, scratch)
    end
end

function run_benchmark(timeloop, state, dt, interval, scratch = void; hours = 6)
    N = max(1, Int(hours * 3600 / interval))
    nstep = round(Int, interval / dt)
    @info "$(N*nstep) steps of length $dt"
    run_loop(timeloop, N, interval, state, scratch; dt)
end

function time(fun, N)
    fun()
    times = [(@timed fun()).time for _=1:2N]
    sort!(times)
    return round(sum(times[1:N])/N; digits=6), round(sum(times)/2N; digits=6) 
end

function show_time(fun, N, nt, single)
    elapsed, _ = time(fun, N)
    if single==0 
        single = elapsed
        @time fun()
    end
    println("$nt \t $elapsed \t $(round(single/elapsed; digits=3)) $(round(100*single/elapsed/nt))%")
    return single
end

function scaling_tendencies(models, u0, N=20)
    du = deepcopy(u0)
    single = 0.0
    for nt in eachindex(models) 
        model = models[nt]
        scratch = scratch_space(model, u0, 0.0)
        single = show_time(N, nt, single) do
            tendencies!(du, scratch, model, u0, 0.0)
        end
    end
end

function scaling_RK4(models, u0, N=3)
    single = 0.0
    for nt in eachindex(models) 
        model = models[nt]
        scheme = RungeKutta4(model)
        solver = IVPSolver(scheme, 1.0, u0, 0.0)
        u = deepcopy(u0)
        single = show_time(N, nt, single) do
            for _ in 1:15
                advance!(u, solver, u, 0.0, 1)
            end
        end
    end
end

#=
using LoopManagers.ManagedLoops: ManagedLoops, LoopManager, @loops, @vec

@loops function complex_copyto2!(_, a, bc, ax1, ax2, ax3)
    let (irange, jrange) = (ax1, ax2)
        for j in jrange, k in ax3
            for i in irange
                @inbounds a[i,j,k] = bc[i,j,k]
            end
        end
    end
end

@loops function complex_copyto3!(_, a, bc, ax1, ax2, ax3) # same implementation as ManagedLoops
    let (irange, jrange, krange) = (ax1, ax2, ax3)
        for j in jrange, k in krange
            for i in irange
                @inbounds a[i,j,k] = bc[i,j,k]
            end
        end
    end
end

@loops function real_copyto4!(_, a, bc, ax1, ax2, ax3, ax4)
    let (irange, jrange, krange) = (ax1, ax2, ax3)
        for j in jrange, k in krange, l in ax4
            @vec for i in irange
                @inbounds a[i,j,k,l] = bc[i,j,k,l]
            end
        end
    end
end

# hijack managed broadcast for 3D complex arrays
function ManagedLoops._internals_.managed_copyto!(mgr::LoopManager, a::Array{<:Complex, 3}, bc, ax1, ax2, ax3)
    complex_copyto2!(mgr, a, bc, ax1, ax2, ax3)
end

# hijack managed broadcast for 4D real arrays
function ManagedLoops._internals_.managed_copyto!(mgr::LoopManager, a::Array{<:Real, 3}, bc, ax1, ax2, ax3, ax4)
    real_copyto4!(mgr, a, bc, ax1, ax2, ax3, ax4)
end
=#
