# ClimFlowsExamples

ClimFlowsExamples is a repository that demonstrates how to compose elements of the [Climflows](https://github.com/ClimFlows) ecosystem into runnable "applications". 

The directory `WIP` contains work-in-progress examples. These may be working, but it is not guaranteed.
Other directories contains examples supposed to work out-of-the box after reading our [general instructions](#installation) and project-specific instructions. Open an issue if this is not the case.

Each example directory is an independent Julia project, with its own `Project.toml` including `compat` entries. This should guarantee that they do not need to follow updates of the ecosystem to continue working. This also means that early applications may reflect an early state of the ecosystem.

## Rotating shallow-water equations, spherical harmonics

https://github.com/ClimFlows/.github/assets/24214175/4410dfe0-eff4-4b8c-b17b-546103ba6579

## Rotating shallow-water equations, mimetic finite differences

https://github.com/ClimFlows/.github/assets/24214175/3ae1b0a0-bef2-4ef1-8602-7c1f86b381a4

## Installation

### On a personal computer
#### Install Julia
The recommended method to install Julia is via 
[juliaup](https://github.com/JuliaLang/juliaup). See also [Modern Julia Workflows](https://modernjuliaworkflows.org/).

#### Install ClimFlowsExamples

In the shell, `cd` to some place where you can store source code and run it. Then:
```shell
git clone https://github.com/ClimFlows/ClimFlowsExamples.git
# or
git clone git@github.com:ClimFlows/ClimFlowsExamples.git
# then
cd ClimFlowsExamples/SpectralRSW.jl
```
You may of course `cd` into another project.
The main program is a `.jl` file with the same name as the directory, here `SpectralRSW.jl`. Before running in, you must install dependencies:
```shell
julia install.jl
```
Installing and precompiling dependencies may take a moment depending on their number and heaviness, but it needs to be done only once.
 
#### Run

Then:
```shell
julia SpectralRSW.jl
ls -lrth # to see created files, if any
```
Voilà !

### On a HPC cluster

Supercomputers usually have the following limitations:
- strict filesystem quota, with different quota on different filesystems; `HOME`, `WORK`...
- login node with internet access and compute node without internet access
- different hardware on login nodes and compute nodes

These limitations can be handled by setting certain [environment variables](https://docs.julialang.org/en/v1/manual/environment-variables/) and doing installation on login nodes and execution on compute nodes. See also [Julia on HPC Clusters](https://juliahpc.github.io/)

#### Installation

Let us start with installation, on a login node. If `julia` is available as a pre-installed `module`, using it is preferable as it will get easier to get support from your
center in case of issues. The following shell commands will list available modules:

```shell
module avail
# or
module avail julia
```
If you have identified a module providing julia, load it:
```shell
module load julia/1.10.4 # for example
```
Julia needs to store many files in `$HOME/.julia` unless the variable `JULIA_DEPOT_PATH` is set. Since `$HOME` is usually on a safe filesystem that comes with very limited quotas, it is almost certainly necessary to set `JULIA_DEPOT_PATH` to a directory that you own on a file system with less strict quota, e.g. :

```shell
# assuming WORK points to a place you own with sufficient quota
export JULIA_DEPOT_PATH="$WORK/julia_depot_path"
```
*`JULIA_DEPOT_PATH` must be accessible from computing nodes !*

The contents of `JULIA_DEPOT_PATH` can be regenerated, i.e. it is possible to entirely delete it with `rm -rf`. You will need to re-install dependencies (see below).

By default julia precompiles dependencies immediately after installing them. Since we install on a login node but execute on a compute node with possibly different hardware, this is useless. Set:
```shell
export JULIA_PKG_PRECOMPILE_AUTO=0
```
to disable automatic precompilation. 

The `module load` and `export ...` commands take effect only in the current shell session. You probably want to put them in a shell script that
you will source when you need `julia`.

You can now [install ClimFlowsExamples](#install-climflowsexamples)

#### Execution

We assume here that you can login interactively onto a compute node. The same shell commands can also be put in a batch job submitted to the job scheduler (e.g. to SLURM via `sbatch`).

First execute the appropriate `module load` and `export ...` commands, or source a shell script where you have stored these commands. Then `cd` to the project directory, e.g. `SpectralRSW.jl`.

We let us know julia that the compute node is airgapped by:
```shell
export JULIA_PKG_OFFLINE=true
```

Since we have not precompiled on the login node, we should do it now (once):
```shell
julia --project=. -e "using Pkg; Pkg.precompile()"
```

You can now [run](run) the project. For full reproducibility, you may prefer:
```shell
julia --startup-file=no SpectralRSW.jl
```
This skips loading your base julia environment and ensures that the program runs exactly in the environment described by `Project.toml`.
