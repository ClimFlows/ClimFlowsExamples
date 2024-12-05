choices = (
    Fluid = IdealPerfectGas,
    TimeScheme = KinnmarkGray{2,5},
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
    hyperdiff_n = 2,
    remap_period = 6,
    nlat = 96,
    ndays = 10,
)
params = (
    ptop = 225.52395239472398,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    hyperdiff_nu = 0, # 0.002,
    courant = 4.0,
    interval = 6 * 3600, # 6-hour intervals
)
