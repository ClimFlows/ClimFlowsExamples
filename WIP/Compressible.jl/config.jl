choices = (Fluid=IdealPerfectGas,
           TimeScheme=ARK_TRBDF2, # ARK_TRBDF2, # Midpoint, #BackwardEuler, # KinnmarkGray{2,5},
           consvar=:temperature,
#           TestCase=Jablonowski06,
           TestCase=DCMIP{21},
           Prec=Float64,
           nz=30,
           hyperdiff_n=3,
           remap_period=0,
           nlat=64,
           newton=(niter=2,         # number of Newton iterations
                   flip_solve=false, # direction of tridiagonal solver passes  
                   update_W=true,   # update W during Newton iterations
                   verbose=false))
params = (
          testcase = (), # (; u0=0, up=1, lonc=pi, latc=0), # override test case defaults
          ptop = 3281.8, # 225.52395239472398,
          pb=1e5,
          rhob=1e5, # 100.0,
          gravity = 9.81/500,
          Phis=0,
          Cp=1000,
          kappa=2 / 7,
          p0=1e5,
          T0=300,
          radius=6.4e6,
          Omega=0, # 7.272e-5,
          hyperdiff_nu=0, # 0.002,
          courant=2.8,
          dt=1000,
          ndays=40,
          interval=3600*24)
