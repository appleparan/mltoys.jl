using Mise
using Test

using DataFrames
using StatsBase: mean, std, mean_and_std, zscore
using Dates, TimeZones
using JuliaDB
using Flux
using CuArrays

CuArrays.allowscalar(false)

include("data.jl")
include("utils.jl")
include("evaluation.jl")
