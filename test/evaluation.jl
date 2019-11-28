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