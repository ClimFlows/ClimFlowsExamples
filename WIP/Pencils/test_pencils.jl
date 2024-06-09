# Usage:
# srun -A wuu@cpu --ntasks=30 --hint=nomultithread julia test_pencils.jl

using Pkg;
Pkg.activate(@__DIR__);

using InteractiveUtils
using MPI
using PencilArrays
using PencilArrays.Transpositions: Transposition
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, void

function setup(nlat, nz)
    MPI.Init()
    comm = MPI.COMM_WORLD
    is_master = MPI.Comm_rank(comm)==0
    is_master && versioninfo()    

    is_master && @info "Initializing MPI..."
    nlon = 2nlat
    horiz = nlon * nlat
    topo = MPITopology(comm, Val(1))

    permute = Permutation(2,1)
    pencil1 = Pencil((horiz, nz), (2,), comm)
    pencil2 = Pencil(pencil1; decomp_dims=(1,))

    if is_master
        @info "Initializing spherical harmonics..."
        @time sph = SHTnsSphere(nlat)
    else
        sph = SHTnsSphere(nlat)
    end
    return comm, pencil1, pencil2, sph
end

function time(fun, N)
    fun()
    times = [(@timed fun()).time for _ in 1:N+10]
    sort!(times)
    return round(sum(times[1:N])/N ; digits=7)
end

function main()
    nlat, nz = 128, 40
    comm, p, q, sph = setup(nlat, nz)

    is_master = MPI.Comm_rank(comm)==0

    F=Float64
    a = Array{F}(undef, size_local(p))
    b = Array{F}(undef, size_local(q))

    trans = Transposition(PencilArray(p, a), PencilArray(q, b))
    
    elapsed = time(10) do
        transpose!(trans)
    end
    is_master && @info "Time for transposition: $elapsed"
    
    aa = reshape(a, nlat, 2nlat, :)
    a_spec = analysis_scalar!(void, aa, sph)
    is_master && @info "sizes" size(b) size(aa) size(a_spec)

    elapsed = time(10) do
        analysis_scalar!(a_spec, aa, sph)
    end
    is_master && @info "Time for spectral transform: $elapsed"
end

main()
