using Pkg;
cuda_dir = joinpath(@__DIR__, "CUDA")
Pkg.activate(Base.active_project() == joinpath(cuda_dir, "Project.toml") ? cuda_dir : joinpath(@__DIR__, "CPU"))
Pkg.Registry.add() # makes sure we do not hide the General registry
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ClimFlows/JuliaRegistry.git"))
Pkg.instantiate()
