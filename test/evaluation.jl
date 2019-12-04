@testset "Generic Evaluation" begin
    @testset "Generic Evaluation - Vector" begin
        in = 5
        x = ones(Int, in)
        y = ones(Int, in)
        ŷ = ones(Int, in)
        ri = abs(rand(Int)) % in + 1
        ŷ[ri] = 10

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
        res = evaluation(dataset, model, statvals, _maxfunc)
        @test Int(res) == 9
        res = evaluation(dataset, model, statvals, _minfunc)
        @test Int(res) == 0

        # multiple metric
        res = evaluations(dataset, model, statvals, [_maxfunc, _minfunc])
        @test Int.(res) == (9, 0)
    end

     @testset "Generic Evaluation - Matrix" begin
        in = 5
        batch = 7
        x = ones(Int, in, batch)
        y = ones(Int, in, batch)
        ŷ = ones(Int, in, batch)
        for i in 1:batch
            ri = abs(rand(Int)) % in + 1
            ŷ[ri, i] = 10
        end

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
        @test _maxfunc(y, ŷ) == reshape(ones(Int, batch) * 9, 1, batch)
        @test _minfunc(y, ŷ) == reshape(ones(Int, batch) * 0, 1, batch)

        # single metric
        res = evaluation(dataset, model, statvals, _maxfunc)
        @test Int(res) == 9
        res = evaluation(dataset, model, statvals, _minfunc)
        @test Int(res) == 0

        # multiple metric
        res = evaluations(dataset, model, statvals, [_maxfunc, _minfunc])
        @test Int.(res) == (9, 0)
    end
end

@testset "RSR" begin
    @testset "best case (should be zero)" begin
        n = 100
        a = rand(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = a
        @test isapprox(evaluation([(a, a)], model, statvals, :RSR), 0.0)
    end
    
    @testset "worst case (should be one)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = b
        @test isapprox(evaluation([(a, b)], model, statvals, :RSR), 0.0)
    end
end

@testset "NSE" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = a
        @test isapprox(evaluation([(a, a)], model, statvals, :NSE), 1.0)
    end 
end

@testset "IOA" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = a
        @test isapprox(evaluation([(a, a)], model, statvals, :IOA), 1.0)
    end
    
    @testset "worst case (should be zero)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = b
        @test isapprox(evaluation([(a, b)], model, statvals, :IOA), 0.0)
    end
end

@testset "R2" begin
    @testset "best case (should be one)" begin
        n = 100
        a = rand(n)
         statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = a
        @test isapprox(evaluation([(a, a)], model, statvals, :R2), 1.0)
    end 
    
    @testset "worst case (should be zero)" begin
        n = 100
        a = ones(n)
        b = zeros(n)
        statvals = ndsparse((
            dataset = ["total", "total", "total", "total"],
            type = ["μ", "σ", "maximum", "minimum"]),
            (value = [mean(a), std(a), 1.0, 0.0],))
        model(_x) = b
        @test isapprox(evaluation([(a, b)], model, statvals, :R2), 0.0)
    end
end
