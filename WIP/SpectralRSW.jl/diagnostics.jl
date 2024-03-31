function vorticity(model, state)
    (; sph) = model
    zeta = curl!(void, state.uv_spec, sph)
    @. zeta = zeta * model.radius^-2
    synthesis_scalar!(void, zeta, sph)
end

function velocity(model, state)
    (; sph, radius) = model
    uv_spec = vector_spec(
        (@. state.uv_spec.spheroidal/radius),
        (@. state.uv_spec.toroidal/radius))
    synthesis_vector!(void, uv_spec, sph)
end

function geopotential(model, state)
    (; radius, sph) = model
    radius^(-2)*synthesis_scalar!(void, state.gh_spec, sph)
end

function potential_vorticity(model, geopotential, vorticity)
    (; fcov, radius) = model
    @. ((radius^-2)*fcov+vorticity)/geopotential
end


diagnostics() = CookBook(; velocity, geopotential, vorticity, potential_vorticity)
