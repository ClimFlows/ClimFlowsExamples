#== Voronoi ==#

function tendencies_SW!( (ducov,dgh), (ucov,gh), (qv, qe, U, B), model, mesh::VoronoiSphere)
    hodges = mesh.le_de
    radius = model.planet.radius
    massflux!(U, ucov, gh, radius, mesh.edge_left_right, hodges)
    bernoulli!(B, gh, ucov, radius, mesh.primal_deg, mesh.Ai, hodges, mesh.primal_edge)
    BK.voronoi_potential_vorticity!(qv, model.fcov, ucov, gh, mesh.Av, mesh.dual_vertex, mesh.dual_edge, mesh.dual_ne, mesh.Riv2)
    BK.voronoi_du!(ducov, qe, qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    BK.voronoi_dm!(dgh, U, mesh.Ai, mesh.primal_deg, mesh.primal_edge, mesh.primal_ne)
end

function tendencies_SW( (ucov,gh), model, mesh::VoronoiSphere)
    hodges = mesh.le_de
    radius = model.planet.radius
    U = massflux(ucov, gh, radius, mesh.edge_left_right, hodges)
    B = bernoulli(gh, ucov, radius, mesh.primal_deg, mesh.Ai, hodges, mesh.primal_edge)
    qv = BK.voronoi_potential_vorticity(model.fcov, ucov, gh, mesh.Av, mesh.dual_vertex, mesh.dual_edge, mesh.dual_ne, mesh.Riv2)
    ducov = BK.voronoi_du(qv, U, B, mesh.edge_down_up,
        mesh.edge_left_right, mesh.trisk_deg, mesh.trisk, mesh.wee)
    dgh = BK.voronoi_dm(U, mesh.Ai, mesh.primal_deg, mesh.primal_edge, mesh.primal_ne)
    return ducov, dgh
end

#== Voronoi, mass flux ==#

function massflux(ucov, m, radius, left_right, hodges)
    U = similar(ucov)
    massflux!(U, ucov, m, radius, left_right, hodges)
    return U
end

function massflux!(U::AbstractVector, ucov, m, radius, left_right, hodges)
    @fast for ij in eachindex(hodges)
        left, right = left_right[1,ij], left_right[2,ij]
        U[ij] = inv(2*radius*radius)*(m[left]+m[right])*ucov[ij]*hodges[ij]
    end
end

massflux!(backend, U::AbstractMatrix, args...) = massflux_3D!(backend, U, args...)

@loops function massflux_3D!(_, U, ucov, m, radius, left_right, hodges)
    let range = axes(U,2)
        @fast for ij in range
            nz = size(U,1)
            left, right = left_right[1,ij], left_right[2,ij]
            for k=1:nz
                U[k,ij] = inv(2*radius*radius)*(m[k,left]+m[k,right])*ucov[k,ij]*hodges[ij]
            end
        end
    end
end

#== Voronoi, kinetic energy ==#

function bernoulli(gh, ucov, radius, degree, areas, hodges, edges)
    B = similar(gh)
    bernoulli!(B, gh, ucov, radius, degree, areas, hodges, edges)
    return B
end

function bernoulli!(B::AbstractVector, gh, ucov, radius, degree, areas, hodges, edges)
    inv_r2 = inv(radius*radius)
    @fast for ij in eachindex(areas)
        deg, inv_area = degree[ij], inv(4*areas[ij])
        @unroll deg in 5:7 B[ij] = inv_r2 * (
            gh[ij] + inv_area*sum(hodges[edges[e,ij]]*ucov[edges[e,ij]]^2 for e=1:deg) )
    end
end

bernoulli!(backend, B::AbstractMatrix, args...) = bernoulli_3D!(backend, B, args...)

@loops function bernoulli_3D!(_, B, ucov, radius, degree, areas, hodges, edges)
    let range = axes(B,2)
        nz = size(B,1)
        @fast for ij in range
            deg, aa = degree[ij], inv(4*areas[ij]*radius*radius)
            @unroll deg in 5:7 begin
                ee = (edges[e,ij] for e=1:deg)
                hh = (hodges[ee[e]] for e=1:deg)
                @simd for k=1:nz
                    B[k,ij] += aa * sum(hh[e]*ucov[k,ee[e]]^2 for e=1:deg)
                end
            end
        end
    end
end
