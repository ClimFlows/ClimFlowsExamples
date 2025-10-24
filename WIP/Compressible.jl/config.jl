choices = (Fluid=IdealPerfectGas,
           TimeScheme=ARK_TRBDF2, # ARK_TRBDF2, # Midpoint, #BackwardEuler, # KinnmarkGray{2,5},
           consvar=:temperature,
           precision=Float32,
           nz=30,
           hyperdiff_n=3,
           remap_period=0,
           nlat=192,
           meshname="uni.2deg.mesh.nc",
           etopo="etopo40.nc",
           newton=(niter=2,          # number of Newton iterations
                   flip_solve=false, # direction of tridiagonal solver passes (`true` no yet implemented in batched HEVI solver)
                   update_W=true,    # update W during Newton iterations (ignored by batched HEVI solver)
                   verbose=false))
params = (testcase=(; Phi_m=(250*9.81)), # xi_m=0.1,
          wfactor=1.0, # 1e4, # exaggerate vertical velocity in initial condition
          Xfactor=500.0, # divide gravity and surface geopotential by Xfactor
          Zfactor=5.0, # multiply surface geopotential by Zfactor
          ptop=225.52395239472398,
          rhob=1e6,
          gravity=9.81,
          Cp=1000,
          kappa=2/7,
          p0=1e5,
          T0=300,
          radius=6.4e6,
          Omega=7.272e-5,
          hyperdiff_nu=0, # 0.002,
          courant=2.5,
          interval=3600*4,
          ndays=20,
          # for tests and benchmarking
          dt=1000,
          Phis=0,
          pb=1e5)

function exp_Jablonowski06(choices, params; Xfactor=params.Xfactor)
    quicklook(session) = plotmap(session.surface_pressure .- 1e5)
    return override(choices; TestCase=Jablonowski06, quicklook),
           override(params; gravity=(params.gravity/Xfactor), ndays=4)
end

function exp_DCMIP21(choices, params; Xfactor=params.Xfactor)
    return override(choices; TestCase=DCMIP{21}, quicklook),
           override(params; ptop=3281.8, Omega=0, gravity=(params.gravity/Xfactor), ndays=15)
end

function exp_DCMIP21_custom(choices, params; Xfactor=params.Xfactor)
    return override(choices; TestCase=DCMIP21_custom, quicklook),
           override(params; ptop=3281.8, Omega=0,
                    gravity=(params.gravity/Xfactor))
end

function quicklook(t, session, interp)
    slice(x::Matrix, lev) = interp(x[lev, :])
    slice(x::Array{3}, lev) = x[:, :, lev]
    lev = 20
    # ps = session.surface_pressure

    T = session.temperature[lev,:,:]
    T = T .- mean(T)
    plotmap(T, "T-<T> at level $lev")

    T = session.temperature
    T = T[1:25,div(size(T,2),2),1:2:div(size(T,3),2)]
    T = T .- mean(T)
    plotmap(T, "T-<T> at the Equator")

    # @info "quicklook" size(T)
    # plotslice(T[:,:,1:end-10])
    # mass = session.state.mass_air
    # mm = mean(mass[lev, :])
    # mass = slice(mass, lev)
    # logmass = @. log10(abs(mass-mm)+1)
    # plotmap(logmass, "Air mass at t=$t s, level $lev")
    # @info "mean mass at t=$t s, level $lev" mm
    # dPhi = session.state.Phi[2:end, :]-session.state.Phi[1:(end - 1), :]
    # plotmap(slice(dPhi, lev), "Layer thickness at t=$t s, level $lev")
    # @info "symmetry" sym(ps,-) sym(Phis,-) sym(T10,-)
end

# experiment(choices, params) = exp_Jablonowski06(choices, params; Xfactor=100)
experiment(choices, params) = exp_DCMIP21(choices, params)
# experiment(choices, params) = exp_DCMIP21(choices, params; Xfactor=500)
# experiment(choices, params) = exp_DCMIP21_custom(choices, params)
