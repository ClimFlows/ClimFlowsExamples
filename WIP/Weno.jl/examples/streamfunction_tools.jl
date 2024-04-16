include("poisson.jl")


alloc_like(x) = x .* 0

function get_circular_domain(shape; nh = 3)
    msk = get_closed_domain(shape; nh = nh)
    nx, ny = shape
    dx = 1.0 / (nx - 2 * nh)
    dy = 1.0 / (ny - 2 * nh)

    for j = nh+1:ny-nh, i = 1+nh:nx-nh
        x = (i - nh - 0.5) * dx
        y = (j - nh - 0.5) * dy
        d2 = (x - 0.5)^2 + (y - 0.5)^2
        r2 = 0.5^2
        msk[i, j] = d2 < r2 ? 1 : 0

    end

    msk[1:nh, :] .= 0
    msk[end-nh:end, :] .= 0

    return msk
end

function get_closed_domain(shape; nh = 3)
    msk = zeros(UInt8, shape)
    @. msk[nh+1:end-nh, nh+1:end-nh] = 1

    return msk
end


function get_body_rotation(msk, dx, dy)
    mskv = get_mask_at_corners(msk)
    F = typeof(dx)
    vorticity = zeros(F, size(msk))
    @. vorticity = 4 * pi * mskv * dx * dy
    psi = get_psi_from_vorticity(vorticity, mskv)
    U, V = get_velocity_from_psi(psi, msk)
    return U, V
end

function get_analytical_flow(msk, dx, dy; nh = 3)
    @assert all(msk[nh+1:end-nh, nh+1:end-nh] .== 1)
    nx, ny = size(msk)
    F = typeof(dx)
    U = zeros(F, nx, ny)
    V = zeros(F, nx, ny)

    for j = 1:ny, i = 2:nx
        x = pi * F(i - nh - 1) * dx
        y = pi * F(j - nh - 0.5) * dy
        U[i, j] = -F(msk[i-1, j] * msk[i, j]) * (sin(x) * cos(y))
    end
    for j = 2:ny, i = 1:nx
        x = pi * F(i - nh - 0.5) * dx
        y = pi * F(j - nh - 1) * dy
        V[i, j] = F(msk[i, j] * msk[i, j-1]) * (cos(x) * sin(y))
    end

    return U, V
end

function get_mask_at_corners(msk)
    mskv = alloc_like(msk)
    nx, ny = size(msk)
    for j = 2:ny, i = 2:nx
        fourcells = msk[i, j] + msk[i-1, j] + msk[i, j-1] + msk[i-1, j-1]
        mskv[i, j] = (fourcells == 4 ? 1 : 0)
    end
    return mskv
end

function get_psi_from_vorticity(vorticity, mskv)
    nx, ny = size(vorticity)
    psi = alloc_like(vorticity)

    A = get_A(mskv; bc = :dirichlet)
    mysolve!(mskv, A, vorticity, psi)
    return psi
end

function get_velocity_from_psi(psi, msk)
    nx, ny = size(psi)
    U, V = alloc_like(psi), alloc_like(psi)
    for j = 1:ny-1, i = 2:nx
        U[i, j] = -(psi[i, j+1] - psi[i, j]) * msk[i, j] * msk[i-1, j]
    end
    for j = 2:ny, i = 1:nx-1
        V[i, j] = (psi[i+1, j] - psi[i, j]) * msk[i, j] * msk[i, j-1]
    end
    return U, V
end
