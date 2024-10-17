using Pkg; Pkg.activate(@__DIR__)
using Revise

using LinearAlgebra
using Statistics

includet("meshmodule.jl")
using .MeshModule

const PERMITTEDOPTIONS = Set(["godunov", "slfv"])
const GODUNOV = 1
const SLFV = 2

function mr_slfv(
    m::Mesh,
    rho::MassField,
    mr::MassField,
    fl::Flux,
    iz::Integer,
    iup::Integer,
    iedge::Integer,
    grad3d::MassVector3d,
    speeddt::EdgeVector
    )::Float64
    centervec = m.radvec_i[:,iup]
    edgevec   = m.radvec_e[:,iedge]
    dxvec = -0.5 * (speeddt.u[iedge, iz] * m.ulam_e[:,iedge] + speeddt.v[iedge, iz] * m.uphi_e[:,iedge])
    xtarget = edgevec + dxvec
    mr_int   = mr.F[iup, iz] + dot(xtarget - centervec, grad3d.F[:,iup, iz]) 
    return mr_int
end

function calc_delta(
    m::Mesh,
    rho::MassField,
    conc::MassField,
    fl::Flux,
    delta_rho::MassField,
    delta_conc::MassField,
    iadv::Integer
)
    # Calculate mixing ratio
    mr = mass_field(mesh, rho.nz)
    divide(conc, rho, mr)
    # if SLFV, we need to calculate 
    if iadv == SLFV
        nv = normal_vector(mesh, rho.nz)
        grad3d = mass_vector3d(mesh, rho.nz)
        tang = tangential_vector(mesh, rho.nz)
        rec = edge_vector(mesh, rho.nz)
        
        # calculate the mixing-ratio gradient
        gradient(mesh, mr, grad3d)
        
        # Limit the mixing-ratio gradient to ensure monotonicity
        gradient_limiter!(mesh, mr, grad3d)
        
        # Reconstruct speed-vector*dt from flux*dt
        fill_normal_vector(mesh, nv, fl)
        normal2tangential(mesh, nv, tang)
        speedreconst(mesh, nv, tang, rec)
    end
    
    for ix in 1:m.nx    # Loop on cells
        for iz in 1:rho.nz    # Vertical loop
            delta_conc.F[ix, iz] = 0.0
            delta_rho.F[ix, iz] = 0.0
            for iedge in 1:m.primal_deg[ix]   # Loop on edges
                edge_num = m.primal_edge[iedge, ix]
                nei_num = m.primal_neighbour[iedge, ix]
                ne = m.primal_ne[iedge, ix]

                # mf is the outgoing mass flux from cell ix
                mf = fl.F[edge_num, iz] * ne

                # Calculate the upwind mixing ratio
                if mf > 0     # ix is the upwind cell
                    iup = ix
                else
                    iup = nei_num
                end
                
                if iadv == GODUNOV
                    mrup = mr.F[iup, iz]
                end
                if iadv == SLFV
                    mrup = mr_slfv(m, rho, mr, fl, iz, iup, edge_num, grad3d, rec)
                end
                        
                mf = mf * rho.F[iup]
                delta_rho.F[ix, iz] = delta_rho.F[ix, iz] - mf / m.Ai[ix]
                delta_conc.F[ix, iz] = delta_conc.F[ix, iz] - mf * mrup / m.Ai[ix]

            end
        end
    end

    return
end

function bulge(lat, lon)::Float64
    return ((1.0 + cos(lon)) / 2.0) * cos(lat)
end

function solid_rotation_uv(lat, lon)
    omega = 1.0
    return cos(lat) * omega, 0.0
end

function solid_rotation_sf(lat, lon)
    # omega is the angular speed
    omega = 1.0
    return sin(lat) * omega
end

#========================= Execution ============================#

# advection option (user choice)
# either SLFV or GODUNOV
iadv = SLFV
mesh = create_mesh("mesh.nc")
print(overview(mesh))
print(size(mesh.primal_bounds_lat))

# Initialize or reinitialize density field

rho = mass_field(mesh, 1)
fill_field(mesh, rho, 1.e+23)   # Not 1 to spot eventual errors in * and / by rho
print(overview(rho))

# Initialize or reinitialize density field and store it for evaluation
conc = mass_field(mesh, 1)
fill_field(mesh, conc, bulge)
print(overview(conc))
target = mass_field(mesh, 1)
target.F .= conc.F   # store for evaluation

# Allocate structures
streamfunction = dual_field(mesh, 1)
fl = flux(mesh, 1)
delta_conc = mass_field(mesh, 1)
delta_rho = mass_field(mesh, 1)

# Initialize time
tini = 0.0
tend = 2pi
nsteps = 500
dt = (tend - tini) / nsteps
t = 0.0

ds = create_netcdf_structure(mesh)
print(typeof(ds))
create_mass_field(ds, 
                  "conc", 
                  "tracer concentration",
                  "molecule / m3",
                  "time_counter cell lev")
create_mass_field(ds, 
                  "rho", 
                  "air concentration", 
                  "molecule / m3", 
                  "time_counter cell lev")
                  
add_time_slot(ds, t)
put_mass_field(ds, "rho", rho)
put_mass_field(ds, "conc", conc)

nsteps=2

for i in 1:nsteps
    @time fill_field(mesh, streamfunction, solid_rotation_sf)
    @time fill_flux(mesh, fl, streamfunction, dt)
    calc_delta(mesh, rho, conc, fl, delta_rho, delta_conc, iadv)
    add(conc, delta_conc)
    add(rho, delta_rho)
    print(locate_max(mesh, conc), "\n")
    print(mass(mesh, conc), "\n")
    t = t + dt
    add_time_slot(ds, t)
    put_mass_field(ds, "rho", rho)
    put_mass_field(ds, "conc", conc)
end


print("FINAL CONCENTRATION\n")
print(overview(conc))
println("CONC RMSE: ", std(conc.F .- target.F))
print("FINAL DENSITY\n")
print(overview(rho))
close(ds)