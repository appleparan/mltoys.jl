using DataFrames
using StatsBase: zscore, mean, std, mean_and_std
using Test

using MLToys: standardize!, exclude_elem

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

@testset "Split" begin
    @testset "split_cols" begin
        cols = ["A", "B", "C", "D"]
        target = "A"
        exclude_target = exclude_elem(cols, target)
        @test cols == ["A", "B", "C", "D"]
        @test exclude_target == ["B", "C", "D"]
        
        target = "C"
        exclude_target2 = exclude_elem(cols, target)
        @test exclude_target2 == ["A", "B", "D"]
    end
end
