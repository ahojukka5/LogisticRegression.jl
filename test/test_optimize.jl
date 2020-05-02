using Test, StaticArrays, Random
using LogisticRegression: optimize!

Random.seed!(0)
X = [rand(2) for _ in 1:10]
Y = [x[1] + x[2] > 1.0 ? 1.0 : 0.0 for x in X]

J, w, b = optimize!([1.0, -1.0], 0.0, X, Y, num_iterations=50, learning_rate=3.0)
@test isapprox(w, [4.9046541360009215, 4.340606046548434])
@test isapprox(b, -3.867136029168926)
