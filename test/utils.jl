using DataFrames
using StatsBase: zscore, mean, std, mean_and_std
using Test

using MLToys: standardize!

@testset "Stats" begin
    @testset "zscore" begin
        a = collect(1:5)
        @test mean(a) == 3.0
        @test std(a) == sqrt(sum(abs2.(a .- mean(a)) / (length(a) - 1)))
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame()

        df[:A] = a 
        standardize!(df, "A", "newA")
        @test df[:newA] == zscore(a)
    end
end