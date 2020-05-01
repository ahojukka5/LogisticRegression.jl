using Test

@testset "Test LogisticRegression.jl" begin
    @testset "test_sigmoid.jl" begin
        include("test_sigmoid.jl")
    end
end
