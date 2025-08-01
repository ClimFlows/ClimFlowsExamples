module Tests

using Test
using LinearAlgebra: SymTridiagonal

using ..SingleSolvers: Thomas

function trisolve(A, B, R, fl)
    flip(x) = fl ? x[length(x):-1:1] : x
    TD = SymTridiagonal(flip(B), flip(-A))
    return flip(TD \ flip(R))
end

function test()
    @testset "Single-column solvers" begin
        N=20
        A=[1.5 for _ in 1:N-1] # off-diagonal
        B=[3.5 for _ in 1:N] # diagonal
        R = randn(N)
        for flip in (false, true)
            @test trisolve(A,B,R,flip) ≈ Thomas(A,B,R,flip)
        end
    end
    return nothing
end

end # module
