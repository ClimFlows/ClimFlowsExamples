using Pkg;
Pkg.activate(@__DIR__);

using InteractiveUtils
using MPI
using PencilArrays
using PencilArrays.Transpositions: Transposition
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, void

function setup()
    MPI.Init()
    comm = MPI.COMM_WORLD
    nlat, nz = 64, 30
    nlon = 2nlat
    horiz = nlon * nlat
    topo = MPITopology(comm, Val(1))

    permute = Permutation(2,1)
    pencil1 = Pencil((horiz, nz), (2,), comm)
    pencil2 = Pencil(pencil1; decomp_dims=(1,))

    sph = SHTnsSphere(nlat)
    return comm, pencil1, pencil2, sph
end

function time(fun, N)
    fun()
    times = [(@timed fun()).time for _ in 1:N+10]
    sort!(times)
    return round(sum(times[1:N])/N ; digits=6)
end

comm, p, q, sph = setup()

F=Float64
a = Array{F}(undef, size_local(p))
b = Array{F}(undef, size_local(q))

trans = Transposition(PencilArray(p, a), PencilArray(q, b))

elapsed = time(10) do
    transpose!(trans)
end

@info "Time for transposition: $elapsed"

a_spec = analysis_scalar!(void, a, sph)
elapsed = time(10) do
    analysis_scalar!(a_spec, a, sph)
end
@info "Time for spectral transform: $elapsed"
