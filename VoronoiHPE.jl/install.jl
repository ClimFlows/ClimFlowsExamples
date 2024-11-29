using Pkg;
Pkg.activate(@__DIR__);
Pkg.Registry.add() # makes sure we do not hide the General registry
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ClimFlows/JuliaRegistry.git"))
Pkg.resolve()
Pkg.instantiate()

# we must download Voronoi meshes on the login node
using ClimFlowsData
ClimFlowsData.DYNAMICO_reader(nothing, "")

# tell CUDA to use local toolkit
using CUDA
CUDA.set_runtime_version!(; local_toolkit=true)
