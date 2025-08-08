choices = (Fluid=IdealPerfectGas,
           TimeScheme=ARK_TRBDF2, # ARK_TRBDF2, # Midpoint, #BackwardEuler, # KinnmarkGray{2,5},
           consvar=:temperature,
           Prec=Float64,
           nz=30,
           hyperdiff_n=3,
           remap_period=0,
           nlat=64,
           newton=(niter=3,          # number of Newton iterations
                   flip_solve=false, # direction of tridiagonal solver passes (`true` no yet implemented in batched HEVI solver)
                   update_W=true,    # update W during Newton iterations (ignored by batched HEVI solver)
                   verbose=false))
params = (
          testcase = (),
          ptop = 225.52395239472398,
          rhob=1e5,
          gravity = 9.81,
          Cp=1000,
          kappa=2/7,
          p0=1e5,
          T0=300,
          radius=6.4e6,
          Omega=7.272e-5,
          hyperdiff_nu=0, # 0.002,
          courant=2.8,
          interval=3600*24,
          # for tests and benchmarking
          dt=1000,         
          Phis=0,
          pb=1e5,
          )

function exp_Jablonowski06(choices, params; X=1) 
    quicklook(session) = plotmap(session.surface_pressure.-1e5)
    return override(choices ; TestCase=Jablonowski06, quicklook), 
        override(params; gravity=params.gravity/X, ndays=4)
end

function exp_DCMIP21(choices, params; X=1)
    function quicklook(session) 
        ps = session.surface_pressure
        Phis = session.model.Phis
        T = session.temperature[:,:,:].-300
        T10 = T[:,:,10]
        plotslice(T[:,:,1:end-10])
        plotmap(T10)
        @info "symmetry" sym(ps,-) sym(Phis,-) sym(T10,-)
    end
    return override(choices ; TestCase=DCMIP{21}, quicklook),
            override(params ; ptop = 3281.8, Omega=0, gravity=params.gravity/X, ndays=15)
end

# experiment(choices, params) = exp_Jablonowski06(choices, params; X=100)
experiment(choices, params) = exp_DCMIP21(choices, params; X=500)
