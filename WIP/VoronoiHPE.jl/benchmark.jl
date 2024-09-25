using Pkg; Pkg.activate(@__DIR__)

using ThreadPinning
pinthreads(:cores)

using ManagedLoops: @with, @vec, @unroll
using LoopManagers: VectorizedCPU, MultiThread
using CFDomains: Stencils, VoronoiSphere
using ClimFlowsData: DYNAMICO_reader

using NetCDF: ncread
using BenchmarkTools

choices = (precision=Float32, meshname = "uni.1deg.mesh.nc")
reader = DYNAMICO_reader(ncread, choices.meshname)
sphere = VoronoiSphere(reader; prec = choices.precision)
@info sphere

function f1(mgr, qU, sphere, q, U)
    @with mgr,
    let krange = axes(qU,1)
        @inbounds for ij in axes(qU, 2)
            deg = sphere.trisk_deg[ij]
            @unroll deg in 9:10 begin
                trisk = Stencils.TRiSK(sphere, ij, Val(deg))
                @vec for k in krange
                    qU[k, ij] = trisk(U, q, k)
                end
            end
        end
    end
end

function f2(mgr, qU, sphere, q, U)
    @with mgr,
    let krange = axes(qU,1)
        @inbounds for ij in axes(qU, 2)
            deg = sphere.trisk_deg[ij]
            qU[:, ij] .= 0
            for edge in 1:deg
                trisk = Stencils.TRiSK(sphere, ij, edge)
                @vec for k in krange
                    qU[k, ij] = trisk(qU, U, q, k)
                end
            end
        end
    end
end

qU, q, U = (randn(Float32, 64, length(sphere.le_de)) for _=1:3)
for (fun, vlen, nt) in [(f2, 8, 1), (f1, 8, 1), (f1, 16, 1), (f1, 8, 2), (f1, 8, 3), (f1, 8, 4)]
    mgr = MultiThread(VectorizedCPU(vlen), nt)
    @info "$fun on $mgr"
    @btime $fun($mgr, $qU, $sphere, $q, $U)
end
