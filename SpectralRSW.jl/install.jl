using Pkg;
Pkg.activate(@__DIR__);
Pkg.Registry.add() # makes sure we do not hide the General registry
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ClimFlows/JuliaRegistry.git"))
Pkg.resolve()
Pkg.instantiate()
