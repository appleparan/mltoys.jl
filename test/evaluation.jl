@testset "Generic Evaluation" begin
    @testset "Generic Evaluation - Vector" begin
        in = 5
        x = ones(Int, in)
        y = ones(Int, in)
        ŷ = ones(Int, in)
        ŷ[2] = 10

        # desired result
        # ŷ - y => [0, 9, 0, 0, ...]

        dataset = [(x, y)]
        # dummy funciton & variables
        model(_x) = ŷ
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [0.0, 1.0, 1.0, 0.0],))

        # find maximum per dims
        _maxfunc(y::AbstractVector, ŷ::AbstractVector) = maximum(abs.(y .- ŷ))
        _maxfunc(y::AbstractMatrix, ŷ::AbstractMatrix) = maximum(abs.(y .- ŷ), dims=[1])

        # find minimum per dims
        _minfunc(y::AbstractVector, ŷ::AbstractVector) = minimum(abs.(y .- ŷ))
        _minfunc(y::AbstractMatrix, ŷ::AbstractMatrix) = minimum(abs.(y .- ŷ), dims=[1])

        # check metric function is right
        @test _maxfunc(y, ŷ) == 9
        @test _minfunc(y, ŷ) == 0

        # single metric
        @show typeof(_maxfunc)
        res = evaluation(dataset, model, statvals, _maxfunc)
        @test Int.(res) == 9
        res = evaluation(dataset, model, statvals, _minfunc)
        @test Int.(res) == 0

        # multiple metric
        res = evaluations(dataset, model, statvals, [_maxfunc, _minfunc])
        @test Int.(res) == (9, 0)
    end
end

#=
@testset "RSR" begin
    @testset "best case (should be zero)" begin
        n = 100
        a = rand(n)
        @test RSR(a, a, mean(a)) == 0.0
    end
    
    @testset "worst case (should be one)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        @test RSR(a, b, 0.0) == 1.0
    end
end

@testset "NSE" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
        @test NSE(a, a, mean(a)) == 1.0
    end 
end

@testset "IOA" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
        @test IOA(a, a, mean(a)) == 1.0
    end
    
    @testset "worst case (should be zero)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        @test IOA(a, b, 0.0) == 0.0
    end
end

@testset "R2" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
        @test R2(a, a, mean(a)) == 1.0
    end 

    
    @testset "worst case (should be zero)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        @test R2(a, b, 0.0) == 0.0
    end
end
=#