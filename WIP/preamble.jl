cd(@__DIR__)
import Pkg ; Pkg.activate(@__DIR__) #src
unique!(push!(LOAD_PATH, "$(@__DIR__)/../include")) #src
#!nb import Pkg ; Pkg.activate(@__DIR__)

using InteractiveUtils

const start_time=time()
toc(str)="At t=$(time()-start_time)s : $str"

try_import(mod) = ( mod in keys(Pkg.project().dependencies) ) && @eval import $(Symbol(mod))

@time_imports begin
    try_import("SIMDFunctions")
    try_import("KernelAbstractions")
    import LoopManagers
end

## these functions return a computing backend
## defining functions rather than const variables ensures that we re-initialize backends each time we re-run
plain(args...) = LoopManagers.PlainCPU()
omp(args...) = LoopManagers.MainThread(plain())
threads(args...) = LoopManagers.MultiThread(plain())
SIMD(args...) = LoopManagers.VectorizedCPU(args...)
ompSIMD(args...) = LoopManagers.MainThread(SIMD(args...))
autotune() = LoopManagers.tune()
tSIMD(args...) = LoopManagers.MultiThread(SIMD(args...))
fake(args...) = LoopManagers.FakeGPU(tSIMD(args...))

if (debug) #src
    @eval macro fast(code) #src
        return esc(code) #src
    end #src
else #src
    @eval macro fast(code)
        return esc(quote
            @inbounds $code
        end)
    end
end #src
