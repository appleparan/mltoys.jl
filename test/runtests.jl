using MLToys
using Test

using DataFrames
using StatsBase: mean, std, zscore
using Dates, TimeZones
using Flux

@testset "MLToys" begin

include("utils.jl")
include("evaluation.jl")

end
