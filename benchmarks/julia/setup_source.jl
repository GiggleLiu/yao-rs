using Pkg
source, environment = ARGS
Pkg.activate(environment)
paths = [source; [joinpath(source, "lib", name) for name in readdir(joinpath(source, "lib")) if isfile(joinpath(source, "lib", name, "Project.toml"))]]
Pkg.develop([PackageSpec(path=path) for path in paths])
Pkg.add([PackageSpec(name="BenchmarkTools", version="1"), PackageSpec(name="JSON", version="0.21")])
Pkg.instantiate()
Pkg.precompile()
