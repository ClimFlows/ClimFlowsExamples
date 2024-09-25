# outputs = (:Phi_dot, :Omega, :ulon, :ulat, :dulat, :surface_pressure, :temperature, :pressure, :geopotential)
outputs = (:Omega, :surface_pressure, :temperature)

choices = (
    mgr = MultiThread(VectorizedCPU(16)), # PlainCPU(), # VectorizedCPU(8),
    precision = Float32,
    coordinate = NCARL30, # SigmaCoordinate
    Fluid = IdealPerfectGas,
    TimeScheme = KinnmarkGray{2,5}, # RungeKutta4,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    meshname = "uni.1deg.mesh.nc",
    compare_to_spectral = false,
    nlat = 64, # for the spectral model
    nz = 30,
    niter_gradrot = 2,
    hyperdiff_n = 2,
    remap_period = 4, # number of RK time steps between two remaps
    ndays = 10,
    nstep_dyn = 6,
    periods = 240,
    filename = "VoronoiHPE",
    outputs
)

params = (
    testcase = (), # override default test case parameters
    nu_gradrot = 1e-16,
    hyperdiff_nu = 0.002,
    courant = 4.0,
    ptop = 225.52395239472398, # compatible with NCARL30 vertical coordinate
    Cp = 1004.5,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.27220521664304e-5,
    interval = 6*3600, # 6-hour intervals between saved snapshots
)
