#== Voronoi ==#

function diag_ulonlat(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, gh = state
    lon, lat = domain.lon_i, domain.lat_i
    coslon, sinlon = cos.(lon), sin.(lon)
    coslat, sinlat = cos.(lat), sin.(lat)
    ulon, ulat = similar(lon), similar(lat)
    args = domain.primal_deg, domain.primal_edge, domain.primal_perot_cov, coslon, sinlon, coslat, sinlat
    Domains.primal_lonlat_from_cov!(ulon, ulat, ucov, args...) do ij, ulon_i, ulat_i
        Planets.lonlat_from_cov(ulon_i, ulat_i, lon[ij], lat[ij], planet)
    end
    return ulon, ulat
end

function diag_curlu(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, gh = state
    areas, signs, edges = domain.Av, domain.dual_ne, domain.dual_edge
    curlu = allocate(:dual, domain)
    a = planet.radius # FIXME
    @fast @unroll for ij in eachindex(curlu)
        zeta = sum( ucov[edges[edge,ij]]*signs[edge,ij] for edge=1:3)
        curlu[ij] = inv(a*a*areas[ij])*zeta
    end
    return curlu
end

function diag_divu(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, gh = state
    divu = similar(gh)
    areas, lon, lat = domain.Ai, domain.lon_i, domain.lat_i
    degree, edges, hodges, signs = domain.primal_deg, domain.primal_edge, domain.le_de, domain.primal_ne
    @fast for ij in eachindex(divu)
        a = Planets.scale_factor(lon[ij], lat[ij], planet)
        deg = degree[ij]
        @unroll deg in 5:7 divu[ij] = inv(a*a*areas[ij]) * sum(
                                        signs[e,ij]*ucov[edges[e,ij]]*hodges[edges[e,ij]] for e=1:deg )
    end
    return divu
end

function diag_gh(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, ghcov = state
    lon, lat, gh = domain.lon_i, domain.lat_i, similar(ghcov)
    @fast for ij in eachindex(gh)
        a = Planets.scale_factor(lon[ij], lat[ij], planet)
        gh[ij] = inv(a*a)*ghcov[ij]
    end
    return gh
end

function diag_pv_voronoi(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, gh     = state
    Av, fcov     = domain.Av, model.fcov
    signs, edges = domain.dual_ne, domain.dual_edge
    Aiv, cells   = domain.Riv2, domain.dual_vertex
    pv = allocate_field(:dual, domain, eltype(domain))
    @fast @unroll for ij in eachindex(pv)
        zeta = sum( ucov[edges[edge,ij]]*signs[edge,ij] for edge=1:3 )
        mv  =  Av[ij]*sum( gh[cells[vertex,ij]]*Aiv[vertex,ij] for vertex=1:3 )
        pv[ij] = inv(mv)*(fcov[ij] + zeta)
    end
    return pv
end

function diag_KE(domain::VoronoiSphere, planet::ConformalPlanet, model, state)
    ucov, gh = state
    degrees, edges, hodges = domain.primal_deg, domain.primal_edge, domain.le_de
    areas, lon, lat, KE = domain.Ai, domain.lon_i, domain.lat_i, similar(gh)
    @fast for ij in eachindex(KE)
        a = Planets.scale_factor(lon[ij], lat[ij], planet)
        deg = degrees[ij]
        @unroll deg in 5:7 KE[ij] = inv(2*a*a*areas[ij]) * sum(
                                    hodges[edges[e,ij]]*ucov[edges[e,ij]]^2 for e=1:deg)
    end
    return KE
end

function diagnostics_SW(::VoronoiSphere, diags)
    diags.pv = diag_pv_voronoi
end

#== Generic ==#

function diag_dstate(domain, planet, model, state, scratch)
    dstate  = Models.allocate_state(model)
    Models.tendencies!(dstate, state, scratch, model)
    return dstate
end

function diagnostics(model::AbstractSW, diags=Registry() ; kwargs...)
    diagnostics_SW(model.domain, diags)
    GFRegistries.reset!(diags ;
        model = model,
        ulonlat = diag_ulonlat,
        KE      = diag_KE,
        gh      = diag_gh,
        divu    = diag_divu,
        curlu   = diag_curlu,
        dstate  = diag_dstate,
        dgh     = (domain, planet, model, dstate) -> diag_gh(domain, planet, model, dstate),
        ddivu   = (domain, planet, model, dstate) -> diag_divu(domain, planet, model, dstate),
        dcurlu  = (domain, planet, model, dstate) -> diag_curlu(domain, planet, model, dstate),
        kwargs... )
    return diags
end
