choices = (Fluid=IdealPerfectGas,
           TimeScheme=ARK_TRBDF2, # ARK_TRBDF2, # Midpoint, #BackwardEuler, # KinnmarkGray{2,5},
           consvar=:temperature,
           precision=Float32,
           nz=30,
           vcoord=SigmaCoordinate,
           hyperdiff_n=3,
           remap_period=0,
           nlat=96,
           meshname="uni.2deg.mesh.nc",
           etopo="etopo40.nc",
           newton=(niter=3,          # number of Newton iterations, 3 needed when Xfactor==1
                   flip_solve=false, # direction of tridiagonal solver passes (`true` no yet implemented in batched HEVI solver)
                   update_W=true,    # update W during Newton iterations (ignored by batched HEVI solver)
                   verbose=false))

params = (testcase=(; Phi_m=(250*9.81)), # xi_m=0.1,
          wfactor=1.0, # 1e4, # exaggerate vertical velocity in initial condition
          Xfactor=1000.0, # divide gravity by Xfactor
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
          courant=2.5,   # ignored at this point
          time_step=450.0, 
          interval=3600*4,
          ndays=1,
          # for tests and benchmarking
          dt=1000,
          Phis=0,
          pb=1e5)

function exp_Jablonowski06(choices, params; Xfactor=params.Xfactor)
    return override(choices; TestCase=Jablonowski06, quicklook=quicklook_JW06),
           override(params; gravity=(params.gravity/Xfactor))
end

function exp_DCMIP21(choices, params; Xfactor=params.Xfactor)
    return override(choices; TestCase=DCMIP{21}, quicklook=quicklook_DCMIP21),
           override(params; ptop=3281.8, Omega=0, gravity=(params.gravity/Xfactor))
end

function exp_DCMIP21_custom(choices, params; Xfactor=params.Xfactor)
    return override(choices; TestCase=DCMIP21_custom, quicklook=quicklook_DCMIP21),
           override(params; ptop=3281.8, Omega=0,
                    gravity=(params.gravity/Xfactor))
end

experiment(choices, params) = exp_Jablonowski06(choices, params)
# experiment(choices, params) = exp_DCMIP21(choices, params)
# experiment(choices, params) = exp_DCMIP21_custom(choices, params)
