using Test
using LogisticRegression: propagate!

w = [1.0, 2.0]
b = 2.0

m = 3

x1 = [1.0, 3.0]
x2 = [2.0, 4.0]
x3 = [-1.0, -3.2]

y1 = 1.0
y2 = 0.0
y3 = 1.0

dw = zeros(2)
J, db = propagate!(dw, w, b, [x1, x2, x3], [y1, y2, y3])
@test isapprox(J, 5.801545319394553)
@test isapprox(dw, [0.99845601, 2.39507239])
@test isapprox(db, 0.00145557813678)
