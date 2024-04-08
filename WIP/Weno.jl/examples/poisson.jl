using LinearSolve
using SparseArrays

function get_A(msk; bc = :neumann)
    nx, ny = size(msk)
    N = sum(msk * 1)

    G = zeros(Int64, (nx, ny))
    G[msk.>0] .= range(1, N)

    Nmax = 5 * N
    I = zeros(Int64, (Nmax,))
    J = zeros(Int64, (Nmax,))
    data = zeros(Float64, (Nmax,))

    kc = 0
    for j in range(1, ny), i in range(1, nx)
        if G[i, j] > 0
            diag = 0
            for dd in [(1, 0), (-1, 0), (0, -1), (0, 1)]
                di, dj = dd

                if (
                    (i + di > 0) &
                    (i + di <= nx) &
                    (j + dj > 0) &
                    (j + dj <= ny) &
                    (G[i+di, j+dj] > 0)
                )
                    kc += 1
                    I[kc] = G[i, j]
                    J[kc] = G[i+di, j+dj]
                    data[kc] = 1
                    diag += 1
                end
            end
            kc += 1
            I[kc] = G[i, j]
            J[kc] = G[i, j]
            data[kc] = bc == :neuman ? -diag : -4
        end
    end
    count = kc

    A = sparse(I[1:count], J[1:count], data[1:count], N, N)
end

function mysolve!(msk, A, b, x)
    rhs = b[msk.>0]
    problem = LinearProblem(A, rhs)
    solve(problem)
    x[msk.>0] = solve(problem)
end
