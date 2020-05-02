using Test, StaticArrays
using LogisticRegression: sigmoid

@test isapprox(sigmoid(0.0), 0.5);
@test isapprox(sigmoid(2.0), 0.88079708);
@test isapprox(sigmoid.([0.0, 2.0]), [0.5, 0.88079708])
@test isapprox(sigmoid.(@SVector [0.0, 2.0]), [0.5, 0.88079708])
