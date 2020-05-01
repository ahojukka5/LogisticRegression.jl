module LogisticRegression

using LinearAlgebra

"""
    sigmoid(z)

Compute the sigmoid of z.
"""
function sigmoid(z::Number)
    return 1.0 / (1.0 + exp(-z))
end

function sigmoid(z::AbstractVector)
    return sigmoid.(z)
end

"""
    propagate(w, b, x, y)

Given single sample (x, y), calculate cost and partial derivatives with respect
to w and b.
"""
function propagate(w::AbstractVector, b::Number, x::AbstractVector, y::Number)
    @assert length(w) == length(x)
    z = dot(w, x) + b
    a = sigmoid(z)
    J = -1.0 * (y * log(a) + (1.0 - y) * log(1.0 - a))
    dz = a - y
    dw = x .* dz
    db = dz
    return (J = J, dw = dw, db = db)
end

"""
    propagate(w, b, X, Y)

Given set of samples (X, Y), calculate total cost and partial derivatives with
respect to w and b. X and Y must be iterables with length of m, where m is
number of samples.
"""
function propagate(
    w::AbstractVector,
    b::Number,
    X::AbstractVector{V},
    Y::AbstractVector,
) where {V<:AbstractVector}
    @assert length(X) == length(Y)
    m = length(X)
    @assert m > 0
    J = 0.0
    dw = zeros(length(w))
    db = 0.0
    for (xi, yi) in zip(X, Y)
        result = propagate(w, b, xi, yi)
        J += result.J
        dw += result.dw
        db += result.db
    end
    J /= m
    dw /= m
    db /= m
    return (J = J, dw = dw, db = db)
end

end # module
