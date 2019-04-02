using DataFrames
using StatsBase: zscore, mean_and_std
using Test

@testset "Stats" begin
    @testset "zscore" begin
        a = 1:5
        @test eman_and_std(collect(a)) == [3.0, sqrt(2.0)]
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame()

        df[:A] = a 
        standardize!(df, "A", "newA")
        @test df[:newA] == zscore(a)
    end
end