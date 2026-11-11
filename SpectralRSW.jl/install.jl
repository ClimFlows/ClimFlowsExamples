using Pkg

# Default to CPU when launched without --project=CUDA.
cuda_project = joinpath(@__DIR__, "CUDA", "Project.toml")
Pkg.activate(Base.active_project() == cuda_project ? dirname(cuda_project) : joinpath(@__DIR__, "CPU"))
Pkg.Registry.add() # makes sure we do not hide the General registry
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ClimFlows/JuliaRegistry.git"))
Pkg.instantiate()
cd(@__DIR__)
