using Test

@testset "Test LogisticRegression.jl" begin
    @testset "test_sigmoid.jl" begin
        include("test_sigmoid.jl")
    end
    @testset "test_propagate.jl" begin
        include("test_propagate.jl")
    end
end
