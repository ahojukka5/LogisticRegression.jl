using Test, StaticArrays
using LogisticRegression: propagate

w = @SVector [1.0, 2.0]
b = 2.0

m = 3

x1 = @SVector [1.0, 3.0]
x2 = @SVector [2.0, 4.0]
x3 = @SVector [-1.0, -3.2]

y1 = 1.0
y2 = 0.0
y3 = 1.0

J1, dw1, db1 = propagate(w, b, x1, y1)
J2, dw2, db2 = propagate(w, b, x2, y2)
J3, dw3, db3 = propagate(w, b, x3, y3)

J = 1/m * (J1 + J2 + J3)
dw = 1/m * (dw1 + dw2 + dw3)
db = 1/m * (db1 + db2 + db3)
@test isapprox(J, 5.801545319394553)
@test isapprox(dw, [0.99845601, 2.39507239])
@test isapprox(db, 0.00145557813678)

J, dw, db = propagate(w, b, [x1, x2, x3], [y1, y2, y3])
@test isapprox(J, 5.801545319394553)
@test isapprox(dw, [0.99845601, 2.39507239])
@test isapprox(db, 0.00145557813678)
