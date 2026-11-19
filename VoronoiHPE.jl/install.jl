using Pkg;
const cuda_project = joinpath(@__DIR__, "CUDA", "Project.toml")
const use_cuda = Base.active_project() == cuda_project
Pkg.activate(use_cuda ? dirname(cuda_project) : joinpath(@__DIR__, "CPU"));
Pkg.Registry.add() # makes sure we do not hide the General registry
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ClimFlows/JuliaRegistry.git"))
Pkg.resolve()
Pkg.instantiate()

# we must download Voronoi meshes on the login node
using ClimFlowsData
ClimFlowsData.DYNAMICO_reader(nothing, "")

if use_cuda
    using CUDA
    CUDA.set_runtime_version!(v"12.9")
end
