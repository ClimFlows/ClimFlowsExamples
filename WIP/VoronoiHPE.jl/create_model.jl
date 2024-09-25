model, diags, state0 = let params = rmap(choices.precision, params)
    reader = DYNAMICO_reader(ncread, choices.meshname)
    vsphere = VoronoiSphere(reader; prec = choices.precision)
    setup(vsphere, choices, params)
end;

vsphere = model.domain.layer
@info vsphere

if choices.compare_to_spectral

    # set some flags to zero to compute only certain terms of du/dt
    CFHydrostatics.debug_flags() = (ke = 0, Phi = 1, gradB = 1, CgradExner = 0, qU = 0)

    # compare initial state diagnostics to a spectral reference computation

    # create spectral model
    nthreads = Threads.nthreads()
    @info "Initializing spherical harmonics..."
    (hasproperty(Main, :ssphere) && ssphere.nlat == choices.nlat) ||
        @time ssphere = SHTnsSphere(choices.nlat, nthreads)
    @info ssphere
    model_spec, diags_spec, state0_spec = setup(ssphere, choices, rmap(Float64, params))

    # interpolate from Voronoi mesh to latitudes and longitudes of spectral model
    to_lonlat = let
        to_deg = 180 / pi
        lons, lats = to_deg * ssphere.lon[1, :], to_deg * ssphere.lat[:, 1]
        permute(data) = permutedims(data, (2, 3, 1)) # k=1,i=2,j=3 -> i=2,j=3,k=1
        permute(data::Matrix) = data
        interp =
            ClimFlowsPlots.SphericalInterpolations.lonlat_interp(vsphere, lons, lats)
        permute ∘ interp
    end

    let
        session = open(diags; model, to_lonlat, state = state0)
        session_spec = open(diags_spec; model = model_spec, state = state0_spec)
        for sym in (:dulat, :geopotential) # (:dulat, :dulon)
            data, data_spec = getproperty(session, sym), getproperty(session_spec, sym)
            delta = data - data_spec
            @info "debug_flags() = $(CFHydrostatics.debug_flags())"
            @info "Max discrepancy for $sym: $(maximum(abs, delta))"
            display(heatmap(maximum(abs, delta; dims = 3)[:, :, 1]))
            display(heatmap(permutedims(maximum(delta; dims = 2)[:, 1, :], (2, 1))))
            display(scatterplot(data_spec[:], delta[:]))
        end
    end

    # restore normal computations
    CFHydrostatics.debug_flags() = (ke = 1, Phi = 1, gradB = 1, CgradExner = 1, qU = 1)

else

    to_lonlat = let
        F = choices.precision
        lons, lats = F.(1:2:360), F.(-89:2:90)
        permute(data) = permutedims(data, (2, 3, 1))
        permute(data::Matrix) = data
        interp =
            ClimFlowsPlots.SphericalInterpolations.lonlat_interp(vsphere, lons, lats)
        permute ∘ interp
    end

end

#=
# save initial state to NetCDF
include("save.jl")
save([state0], "$(choices.filename).init.nc") do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end

# benchmark tendencies!

versioninfo()
dstate, scratch_HV = CFHydrostatics.Voronoi.Dynamics.tendencies_HV!(void, void, model, state0, 0);

for mgr in (PlainCPU(), VectorizedCPU(), MultiThread(VectorizedCPU()))
    model_perf = CFHydrostatics.HPE(mgr, model.vcoord, model.planet, model.domain, model.gas, model.fcov, model.Phis);
    @info "tendencies_HV! on $mgr"
    CFHydrostatics.Voronoi.Dynamics.tendencies_HV!(dstate, scratch_HV, model_perf, state0, 0);
    display(@benchmark CFHydrostatics.Voronoi.Dynamics.tendencies_HV!($dstate, $scratch_HV, $model_perf, $state0, 0);)
    @info "tendencies! on $mgr"
    CFHydrostatics.Voronoi.Dynamics.tendencies!(dstate, scratch_HV, model_perf, state0, 0);
    display(@benchmark CFHydrostatics.Voronoi.Dynamics.tendencies!($dstate, $scratch_HV, $model_perf, $state0, 0);)
end

=#
