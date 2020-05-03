using Test
using LogisticRegression: predict

W = ones(1, 1)
b = zeros(1)
y1 = predict(W, b, [-1.0])
y2 = predict(W, b, [ 1.0])
@test y1 == [0]
@test y2 == [1]