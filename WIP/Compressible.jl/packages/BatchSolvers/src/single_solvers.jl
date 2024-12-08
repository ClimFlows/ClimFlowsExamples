module SingleSolvers

# non-batched solvers, mostly for the purpose of testing or reference implementation

# Thomas algorithm for symmetric tridiagonal problem
# Forward sweep can be in increasing order (flip_solve=false) or decreasing order.

Thomas(A, B, R, flip_solve) = flip_solve ? Thomas_flip(A,B,R) : Thomas_noflip(A,B,R)

function Thomas_noflip(A, B, R)
    # Thomas algorithm for symmetric tridiagonal system
    Nz = length(A)
    C, D, x = similar(A), similar(B), similar(R)
    # Forward sweep
    let l = 1
        X = inv(B[l])
        C[l] = -A[l] * X
        D[l] = R[l] * X
    end
    for l in 2:Nz
        X = inv(B[l] + A[l - 1] * C[l - 1])
        D[l] = (R[l] + A[l - 1] * D[l - 1]) * X
        C[l] = -A[l] * X
    end
    let l = Nz + 1
        X = inv(B[l] + A[l - 1] * C[l - 1])
        D[l] = (R[l] + A[l - 1] * D[l - 1]) * X
    end
    # Back-substitution
    x[Nz + 1] = D[Nz + 1]
    for l in Nz:-1:1
        x[l] = D[l] - C[l] * x[l + 1]
    end
    return x
end

function Thomas_flip(A, B, R)
    # Thomas algorithm for symmetric tridiagonal system
    Nz = length(A)
    C, D, x = similar(A), similar(B), similar(R)
    # Forward sweep
    let ll = Nz + 1
        X = inv(B[ll])
        C[ll - 1] = -A[ll - 1] * X
        D[ll] = R[ll] * X
    end
    for ll in Nz:-1:2
        X = inv(B[ll] + A[ll] * C[ll])
        D[ll] = (R[ll] + A[ll] * D[ll + 1]) * X
        C[ll - 1] = -A[ll - 1] * X
    end
    let ll = 1
        X = inv(B[ll] + A[ll] * C[ll])
        D[ll] = (R[ll] + A[ll] * D[ll + 1]) * X
        # Back-substitution
        x[ll] = D[ll]
    end
    for ll in 1:Nz
        x[ll + 1] = D[ll + 1] - C[ll] * x[ll]
    end
    return x
end

end # module
