# ## Preamble
using Revise
using Pkg;
Pkg.activate(@__DIR__);

function Y11(x,y,z,lon,lat)
    θ = π/2 - lat
    ϕ = lon
    return -sqrt(3/(2π)) * real(exp(1im * ϕ)) * sin(θ)
end

function test_azimuthal_phase(sph)
    xy_spat = sample_scalar!(void, Y11, sph)
    xy_spec = analysis_scalar!(void, xy_spat, sph)
    l, m = 1, 1
    LM = (m * (2 * sph.lmax + 2 - (m + 1))) >> 1 + l + 1
    result = xy_spec[LM]
    @test real(result) ≈ 1.0
    @test imag(result) ≈ 0.0 atol=eps()
end

nlat = 128
sph = HarmonicSphere(nlat)
@show sph

test_azimuthal_phase(sph)
