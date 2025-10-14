using ClimFlowsTestCases
import ClimFlowsTestCases: describe, initial
using Interpolations, NetCDF

struct DCMIP21_custom{P} <: ClimFlowsTestCases.TestCaseHPE
    params :: P

    function DCMIP21_custom(F=Float64; Phis, user...) 
        # parameters
        # λ_c        lambda_m
        # ϕ_c        phi_m
        # d          delta_m
        # ξ          xi_m
        # h_0        Phi_m = g h_0
        # p_eq       p0
        # T_eq       pv0 = RT₀
        # u_eq       u0

        # non-generic parameters
        Rd, T0 = 287.0, 300.0 # from DCMIP 2012 document v1.6_23 pp. 4, 30
        p = (p0=1e5, pv0=Rd*T0, u0=20.0, Phis)
        p = rmap(F, override(p, NamedTuple(user)))
        new{typeof(p)}(p)
    end

end

#================= DCMIP2.1 : Mountain waves ===============#

function describe(case::DCMIP21_custom)
    "DCMIP 2.1 test case: orographic wave with custom orography"
end

function initial(case::DCMIP21_custom, lon, lat)
    (; p0, pv0, u0, Phis) = case.params
    Phi = Phis(lon, lat)
    ps = p0*exp(-(Phi + sin(lat)^2*u0^2/2)/pv0)
    return ps, Phi
end

function initial(case::DCMIP21_custom, lon, lat, p)
    (; p0, pv0, u0) = case.params
    Phi = pv0*log(p0/p) - sin(lat)*u0^2/2    
    ulon = u0*cos(lat)
    z = zero(Phi)
    return Phi, ulon, z, z
end

function get_topo(file)
    topographies = Dict(
        "etopo40.nc" => ("ETOPO40X", "ETOPO40Y", "ROSE"),
        "etopo20.nc" => ("ETOPO20X1_1081", "ETOPO20Y", "ROSE"),
        "etopo5.nc" => ("X", "Y", "bath"),
    )
    lon, lat, h = topographies[file]
    to_rad = pi/180
    file = joinpath(@__DIR__, file)
    lon, lat, h = map(n->NetCDF.open(file, n), (lon, lat, h)) 
    lon, lat, h = to_rad*lon[:], to_rad*lat[:], 9.81*h[:,:]
    return lon.-mean(lon), lat.-mean(lat), @. max(0, h)
end

function to_voronoi(lon, lat, data)
    nlat = size(data, 2)
    itp = linear_interpolation(axes(data), data ; extrapolation_bc=(Periodic(), Line()))    
    to_index, minlon = nlat/pi, minimum(lon)
    lon = @. 1+mod(to_index*(lon-minlon), 2nlat)
    lat = @. 1+to_index*(lat+pi/2)
    return itp.(lon, lat)
end
