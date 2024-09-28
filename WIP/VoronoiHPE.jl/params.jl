# choices is for discrete parameters, while params is for continuous parameters (floats)

choices = (
    # computing
    try_gpu = true,
    compare_to_spectral = true,
    cpu = MultiThread(VectorizedCPU(16)), # PlainCPU(), # VectorizedCPU(8),
    precision = Float32,
    # numerics
    meshname = "uni.1deg.mesh.nc",
    coordinate = NCARL30, # SigmaCoordinate
    nz = 30,
    nlat = 64, # for the spectral model
    consvar = :temperature,
    TimeScheme = KinnmarkGray{2,5}, # RungeKutta4,
    remap_period = 4, # number of RK time steps between two remaps
    # physics
    Fluid = IdealPerfectGas,
    TestCase = Jablonowski06,
    # simulation
    ndays = 10,
    filename = "VoronoiHPE",
    outputs = (:Omega, :surface_pressure, :temperature),
    # outputs = (:Phi_dot, :Omega, :ulon, :ulat, :dulat, :surface_pressure, :temperature, :pressure, :geopotential),
)

params = (
    # numerics
    courant = 4.0,
    # physics
    radius = 6.4e6,
    Omega = 7.27220521664304e-5,
    ptop = 225.52395239472398, # compatible with NCARL30 vertical coordinate
    Cp = 1004.5,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    nu_gradrot = 1e-16,
    hyperdiff_nu = 0.002,
    # simulation
    testcase = (), # to override default test case parameters
    interval = 6 * 3600, # 6-hour intervals between saved snapshots
)
