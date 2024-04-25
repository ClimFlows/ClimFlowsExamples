function initialize_SW(domain::VoronoiSphere, model, fun, args...)
    gh, ulon, ulat = allocate_fields( (:scalar, :vector, :vector), domain, eltype(domain) )
    for ij in eachindex(gh)
        ulon_, ulat_, gh[ij] = fun(domain.lon_i[ij], domain.lat_i[ij], args...)
    end
    for ij in eachindex(ulon)
        ulon[ij], ulat[ij], gh_ = fun(domain.lon_e[ij], domain.lat_e[ij], args...)
    end
    return ulon, ulat, gh
end

function observable_to_prognostic(planet::ConformalPlanet, domain::VoronoiSphere, (ulon,ulat,gh))
    de, gh, uv = domain.de, copy(gh), allocate_field(:vector, domain, eltype(domain))
    for ij in eachindex(gh)
        a = Planets.scale_factor(domain.lon_i[ij], domain.lat_i[ij], planet)
        gh[ij] = a*a*gh[ij]
    end
    for ij in eachindex(uv)
        a = Planets.scale_factor(domain.lon_e[ij], domain.lat_e[ij], planet)
        angle = domain.angle_e[ij]
        uv[ij] = (cos(angle)*ulon[ij] + sin(angle)*ulat[ij])*a*de[ij]
    end
    return (uv, gh)
end
