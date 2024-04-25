#== potential vorticity q = curl(ucov)/m ==#

function voronoi_potential_vorticity(fv, ucov, m, areas, cells, edges, signs, weights)
    qv = similar(fv)
    voronoi_potential_vorticity!(qv, fv, ucov, m, areas, cells, edges, signs, weights)
    return qv
end

function voronoi_potential_vorticity!(qv::AbstractVector, fv, ucov, m, areas, cells, edges, signs, weights)
    @fast @unroll for ij in eachindex(qv)
        zeta = sum(ucov[edges[edge,ij]]*signs[edge,ij] for edge=1:3 )
        mv =  areas[ij]*sum(m[cells[vertex,ij]]*weights[vertex,ij] for vertex=1:3 )
        qv[ij] = inv(mv)*(fv[ij]+zeta)
    end
    return qv
end

function voronoi_potential_vorticity!(qv::AbstractMatrix, args...)
    voronoi_PV_mat!(qv, args...)
    return qv!
end

@loops function voronoi_potential_vorticity!(_, qv, fv, ucov, m, areas, cells, edges, signs, weights)
    let (krange, ijrange) = axes(qv)
        @fast @unroll for ij in ijrange
            F = eltype(qv)
            ee = ( edges[edge,ij] for edge=1:3 )
            ss = ( F(signs[edge,ij]) for edge=1:3 )
            cc = ( cells[vertex,ij] for vertex=1:3 )
            ww = ( weights[vertex,ij] for vertex=1:3 )
            @simd for k in krange
                zeta = sum(ucov[k,ee[edge]]*ss[edge] for edge=1:3 )
                mv =  areas[ij]*sum(m[k,cc[vertex]]*ww[vertex] for vertex=1:3 )
                qv[k,ij] = inv(mv)*(fv[ij]+zeta)
            end
        end
    end
end

#== mass tendency dm = -div(U) ==#

function voronoi_dm(U, areas, degree, edges, signs)
    dm = similar(areas)
    voronoi_dm!(dm, U, areas, degree, edges, signs)
    return dm
end

function voronoi_dm!(dm::AbstractVector, U, areas, degree, edges, signs)
    @fast for ij in eachindex(dm)
        deg = degree[ij]
        @unroll deg in 5:7 dm[ij] = -inv(areas[ij]) * sum( signs[e,ij]*U[edges[e,ij]] for e=1:deg )
    end
    return dm
end

@loops function voronoi_dm_3D!(_, dm, U, areas, degree, edges, signs)
    let (krange, ijrange) = (axes(dmass,1),axes(dmass,2))
        F = eltype(dm)
        @fast for ij in ijrange
            deg, aa = degree[ij], inv(areas[ij])
            @unroll deg in 5:7 begin
                ee = ( edges[e,ij] for e=1:deg )
                ss = ( F(signs[e,ij]) for e=1:deg )
                @simd for k in krange
                    dm[k,ij] = -aa * sum( ss[e]*U[k,ee[e]] for e=1:deg )
                end
            end
        end
    end
end

@loops function voronoi_dm_3D!(_, dmass, U, B, areas, degree, edges, signs, left_right)
    let (krange, ijrange) = (axes(dmass,1),axes(dmass,2))
        @views begin
            dm     = dmass[:,:,1]
            dTheta = dmass[:,:,2]
            theta  = B[:,:,2]
            left   = left_right[1,:]
            right  = left_right[2,:]
        end

        F = eltype(dm)
        @fast for ij in ijrange
            deg, aa = degree[ij], inv(areas[ij])
            aa2 = half(aa)
            @unroll deg in 5:7 begin
                ee = ( edges[e,ij] for e=1:deg )
                ss = ( F(signs[e,ij]) for e=1:deg )
                ll = ( left[ee[e]] for e=1:deg )
                rr = ( right[ee[e]] for e=1:deg )
                @simd for k in krange
                    dm[k,ij] = -aa * sum( ss[e]*U[k,ee[e]] for e=1:deg )
                    dTheta[k,ij] = -aa2 * sum( (ss[e]*U[k,ee[e]])*(theta[k,ll[e]]+theta[k,rr[e]]) for e=1:deg )
                end
            end
        end

    end
end

#== velocity tendency du = -q x U - grad B ==#

function voronoi_du(qv, U, B, down_up, left_right, degree, edges, w)
    du, qe = similar(U), similar(U)
    voronoi_du!(du, qe, qv, U, B, down_up, left_right, degree, edges, w)
    return du
end

function voronoi_du!(du::AbstractVector, qe, qv, U, B, down_up, left_right, degree, edges, w)
    # interpolate PV q from v-points (dual cells=triangles) to e-points (edges)
    @fast for ij in eachindex(qe)
        qe[ij] = (1//2)*(qv[down_up[2,ij]]+qv[down_up[1,ij]])
    end
    @fast for ij in eachindex(du)
        deg=degree[ij]
        @unroll deg in 9:12 qV = (1//2)*sum(
            U[edges[e,ij]]*(qe[ij]+qe[edges[e,ij]])*w[e,ij] for e in 1:deg )
        # Remark : the sign convention of wee is such that
        # du = dB + qV, not du = dB-qV
        du[ij] = (B[left_right[1,ij]]-B[left_right[2,ij]]) + qV
    end
    return du
end

voronoi_du!(du::AbstractMatrix, args...) = voronoi_du_3D!(threaded, voronoi_du_wUq_vec!, du, args...)

function voronoi_du_3D!(backend, wUq!::Fun, du, qe, qv, U, B, down_up, left_right, degree, edges, w, N) where Fun
    offload(voronoi_du_qe!, backend, axes(qe), qe, qv, down_up)
    barrier(backend, "voronoi_du_3D!")
    offload(wUq!, backend, axes(du), du, qe, U, B, left_right, degree, edges, w, N)
    return du
end

# Interpolates PV q from v-points (dual cells=triangles) to e-points (edges)
function voronoi_du_qe!((krange, ijrange), qe,qv, down_up)
    @fast for ij in ijrange
        down, up = down_up[1,ij], down_up[2,ij]
        @simd for k in krange
            qe[k,ij] = half(qv[k,up]+qv[k,down])
        end
    end
end

# The following two functions are two slightly different implementations of du = -q x U - grad(B)
# where q has been interpolated to e-points (edges) by voronoi_du_qe!.
# One of these functions is passed to voronoi_du_3D! as the argument wUq!

# In this implementation, the local loop over edges is not unrolled. Instead it is
# made the outer loop, so that the inner loop is w.r.t the vertical index, and vectorizes.
@inline function voronoi_du_wUq_vec!((krange, ijrange)::Tuple, du, qe,U,B, left_right, degree, edges, w, N)
    voronoi_du_wUq_vec!(ijrange, krange, du, qe,U,B, left_right, degree, edges, w, N)
end

function voronoi_du_wUq_vec!(ijrange, krange, du, qe,U,B, left_right, degree, edges, w, N)
    @fast for ij in ijrange
        for k in krange
            du[k,ij] = 0
        end
        deg = degree[ij]
        for e = 1:deg
            ww = w[e,ij]
            ee = edges[e,ij]
            @simd for k in krange
                du[k,ij] = muladd(ww*U[k,ee], qe[k,ij]+qe[k,ee], du[k,ij])
            end
        end
        voronoi_du_grad!(ij, krange, du, B, left_right, N)
    end
end

# In this implementation, the loop over vertical indices is the outer loop
# The inner loop over edges is unrolled in batches of 3 edges
# Indeed full unrolling prevents vectorization
function voronoi_du_wUq_unroll!(range, tag, du, qe,U,B, left_right, degree, edges, w, N)
    nz = size(du,1)
    @fast for ij in range
        @simd for k=1:nz
            du[k,ij] = 0
        end
        @unroll for batch=1:3
            off = 3*(batch-1)
            ww = ( w[e+off,ij] for e in 1:3 )
            ee = ( edges[e+off,ij] for e in 1:3 )
            @simd for k=1:nz
                du[k,ij] += sum( (ww[e]*U[k,ee[e]])*(qe[k,ij]+qe[k,ee[e]]) for e in 1:3 )
            end
        end
        # finish the sum if more than 9 edges
        off = 9
        deg = degree[ij]-off
        @unroll deg in 1:3 begin # number of edges = deg+off
            ww = ( w[e+off,ij] for e in 1:deg )
            ee = ( edges[e+off,ij] for e in 1:deg )
            @simd for k=1:nz
                du[k,ij] += sum( (ww[e]*U[k,ee[e]])*(qe[k,ij]+qe[k,ee[e]]) for e in 1:deg )
            end
        end
        voronoi_du_grad!(ij, du, B, left_right, N)
    end
end

@inline function voronoi_du_grad!(ij, krange, du, B::AbstractMatrix, left_right, ::Val{0})
    @fast begin
        left, right = left_right[1,ij], left_right[2,ij]
        @simd for k in krange
            # Remark : the sign convention of wee is such that
            # du = dB + qV, not du = dB-qV
            du[k,ij] = (B[k,left]-B[k,right]) + (1//2)*du[k,ij]
        end
    end
end

@inline function voronoi_du_grad!(ij, krange, du, B::AbstractArray, left_right, ::Val{1})
    @fast begin
        theta = view(B,:,:,2)
        exner = view(B,:,:,3)
        B     = view(B,:,:,1)
        left, right = left_right[1,ij], left_right[2,ij]
        @simd for k in krange
            # Remark : the sign convention of wee is such that
            # du = dB + qV, not du = dB-qV
            dB = B[k,left]-B[k,right]
            dExner = exner[k,left]-exner[k,right]
            theta2 = theta[k,left]+theta[k,right]
            du[k,ij] = dB + half(theta2*dExner + du[k,ij])
        end
    end
end

#== velocity tendency du = -q x U - grad B - theta * grad(B) ==#
