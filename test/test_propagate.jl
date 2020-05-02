using Test
using LogisticRegression: propagate!

w = [1.0 2.0]
b = [2.0]

x1 = [1.0, 3.0]
x2 = [2.0, 4.0]
x3 = [-1.0, -3.2]

y1 = 1.0
y2 = 0.0
y3 = 1.0

J = zeros(1)
dw = zeros(1, 2)
db = zeros(1)
z = zeros(1)
a = zeros(1)
dz = zeros(1)
propagate!(J, dw, db, w, b, [x1, x2, x3], [y1, y2, y3], z, a, dz)

@test isapprox(first(J), 5.801545319394553)
@test isapprox(dw, [0.99845601 2.39507239])
@test isapprox(db, [0.00145557813678])
