
# JULIA SETUP INSTRUCTIONS FOR CHEME 5800 LABS
# This file documents the setup process

println("JULIA SETUP INSTRUCTIONS FOR CHEME 5800 LABS")
println("")
println("Opening Julia Setup for CHEME 5800 Labs")
println("IMPORTANT: Use the lab repository (CHEME-5800-Labs-Fall-2025)")
println("")

# Key setup steps:
# 1. Open GitHub Desktop
# 2. Press Ctrl+~ to open terminal
# 3. Type: julia
# 4. Type: using Pkg
# 5. Type: Pkg.add(\"IJulia\")
# 6. Type: jupyterlab(dir = pwd())

# Example command sequence:
# julia --project=.
# julia> using IJulia
# julia> jupyterlab(dir=@__DIR__())

# To include a file:
include(joinpath(@__DIR__(), "Include.jl"))

# Package status check:
# (L1b) pkg> status

# Manifest File Debugging:
# open terminal on file 
# ls
# cd lectures
# cd week-1
# cd L1c
# julia --project=.
# ]
# activate .
# include("Include,jl") --> usually adds IJulia, should have manifest.toml file created 
# using Pkg
# Pkg.add("IJulia")
# using IJulia
# jupyterlab(dir = pwd())