@testset "Stats" begin
    @testset "zscore" begin
        a = collect(1:5)
        @test mean(a) == 3.0
        @test std(a) == sqrt(sum(abs2.(a .- mean(a)) / (length(a) - 1)))
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame(A = a)
        zscore!(df, "A", "newA")
        @test df[:, :newA] == zscore(a)
    end
end


@info "Testing hampel..."
@testset "hampel" begin
    @testset "hampel_single" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        hampel!(df, "A", "nA")
        # @test isapprox(df[:nA], [0.494191, 0.498064, 0.501936, 0.505809])
    end
end

@testset "zscore" begin
    @testset "zscore_single_string" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, "A", "nA")
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
    end

    @testset "zscore_single_symbol" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, :A, "nA")
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
    end

    @testset "zscore_Array_string" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, ["A", "B"], ["nA", "nB"])
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:, :nB], [ -1.161895003862225,
             -0.3872983346207417,
              0.3872983346207417,
              1.161895003862225])
    end

    @testset "zscore_Array_symbol" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, [:A, :B], [:nA, :nB])
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:, :nB], [ -1.161895003862225,
             -0.3872983346207417,
              0.3872983346207417,
              1.161895003862225])
    end
end
