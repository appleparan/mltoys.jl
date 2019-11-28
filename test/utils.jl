@testset "Stats" begin
    @testset "zscore_stats" begin
        a = collect(1:5)
        @test mean(a) == 3.0
        @test std(a) == sqrt(sum(abs2.(a .- mean(a)) / (length(a) - 1)))
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame(A = a)
        zscore!(df, :A, :newA)
        @test df[!, :newA] == zscore(a)
    end
end

@testset "zscore" begin
    @testset "zscore_df" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, [:A, :B], [:nA, :nB])
        @test isapprox(df[!, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[!, :nB], [ -1.161895003862225,
             -0.3872983346207417,
              0.3872983346207417,
              1.161895003862225])
    end

    @testset "unzscore" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)

        mean_A, std_A = mean_and_std(df[!, :A])
        mean_B, std_B = mean_and_std(df[!, :B])

        zscore!(df, [:A, :B], [:nA, :nB])
        @test isapprox(unzscore(df[!, :nA], mean_A, std_A), df[!, :A])
        @test isapprox(unzscore(df[!, :nB], mean_B, std_B), df[!, :B])
    end
end

@testset "Min-Max Scaling" begin
    @testset "Check bounds" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)

        target_min = 0.0
        target_max = 10.0

        minmax_scaling!(df, [:A, :B], [:nA, :nB], target_min, target_max)

        @test maximum(df[!, :nA]) <= target_max
        @test maximum(df[!, :nB]) <= target_max
        @test minimum(df[!, :nA]) >= target_min
        @test minimum(df[!, :nB]) >= target_min
    end

    @testset "Check bounds" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)

        target_min = 0.0
        target_max = 10.0

        minmax_scaling!(df, [:A, :B], [:nA, :nB], target_min, target_max)

        @test isapprox(unminmax_scaling(df[!, :nA], 1, 4, target_min, target_max), df[!, :A])
        @test isapprox(unminmax_scaling(df[!, :nB], 5, 8, target_min, target_max), df[!, :B])
    end
end
