function initialize_SW(domain::VoronoiSphere, model, fun, args...)
    gh, ulon, ulat = allocate_fields( (:scalar, :vector, :vector), domain, eltype(domain) )
    for ij in eachindex(gh)
        gh[ij], ulon_, ulat_ = fun(domain.lon_i[ij], domain.lat_i[ij], args...)
    end
    for ij in eachindex(ulon)
        gh_, ulon[ij], ulat[ij] = fun(domain.lon_e[ij], domain.lat_e[ij], args...)
    end
    return observable_to_prognostic(model.planet, domain, (; gh, ulon, ulat))
end

function observable_to_prognostic(planet, domain::VoronoiSphere, (; gh, ulon, ulat))
    de, ghcov, ucov = domain.de, copy(gh), allocate_field(:vector, domain, eltype(ulon))
    for ij in eachindex(ghcov)
        a = scale_factor(domain.lon_i[ij], domain.lat_i[ij], planet)
        ghcov[ij] = a*a*gh[ij]
    end
    for ij in eachindex(ucov)
        a = scale_factor(domain.lon_e[ij], domain.lat_e[ij], planet)
        angle = domain.angle_e[ij]
        ucov[ij] = (cos(angle)*ulon[ij] + sin(angle)*ulat[ij])*a*de[ij]
    end
    return (; ghcov, ucov)
end
