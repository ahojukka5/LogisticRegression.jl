using Test
using LogisticRegression: sigmoid

@test isapprox(sigmoid(0.0), 0.5);
@test isapprox(sigmoid(2.0), 0.88079708);
