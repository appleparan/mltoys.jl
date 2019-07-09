using Mise
using Test

using DataFrames
using StatsBase: mean, std, zscore
using Dates, TimeZones
using Flux

@testset "Mise" begin

include("utils.jl")
include("evaluation.jl")

end
