using Test

@testset "Test LogisticRegression.jl" begin
    @testset "test_sigmoid.jl" begin
        include("test_sigmoid.jl")
    end
    @testset "test_propagate.jl" begin
        include("test_propagate.jl")
    end
    @testset "test_optimize.jl" begin
        include("test_optimize.jl")
    end
    @testset "test_predict.jl" begin
        include("test_predict.jl")
    end
end
