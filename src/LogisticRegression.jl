module LogisticRegression

using LinearAlgebra

"""
    sigmoid(z)

Compute the sigmoid of z.
"""
@inline function sigmoid(z::Number)
    return 1.0 / (1.0 + exp(-z))
end

"""
    logistic_loss(y, a)

Calculate logistic loss or cross-entropy loss for sample.
"""
@inline function logistic_loss(y::Number, a::Number)
    return -1.0 * (y * log(a) + (1.0 - y) * log(1.0 - a))
end

"""
    propagage!(dw, w, b, X, Y)

Forward- and backpropagation.

Given set of samples (X, Y), calculate total cost and partial derivatives with
respect to w and b. X and Y must be iterables with length of m, where m is the
number of samples. The first argument `dw` is reused to store partial
derivatives of J with respect weights.
"""
function propagate!(
    dw::AbstractVector,
    w::AbstractVector,
    b::Number,
    X::AbstractVector{V},
    Y::AbstractVector,
) where {V<:AbstractVector}
    @assert length(X) == length(Y)
    m = length(X)
    @assert m > 0
    fill!(dw, 0.0)
    J = 0.0
    db = 0.0
    for (x, y) in zip(X, Y)
        z = dot(w, x) + b
        a = sigmoid(z)
        dz = a - y
        J += logistic_loss(y, a)
     @. dw += x * dz
        db += dz
    end
    J /= m
    @. dw /= m
    db /= m
    return (J=J, db=db)
end

"""
    optimize!(w, b, X, Y; num_iterations = 1000, learning_rate = 1.0)

Given initial model (w, b) and data (X, Y), optimize model parameters using
gradient descent method. Model optimization is done in place. Returns optimized
parameters and cost for each iteration for convergence studies.
"""
function optimize!(
    w::AbstractVector,
    b::Number,
    X::AbstractVector{T},
    Y::AbstractVector;
    num_iterations::Integer = 1000,
    learning_rate::Float64 = 1.0,
) where {T<:AbstractVector}
    J = zeros(num_iterations)
    dw = zeros(length(w))
    for i in 1:num_iterations
        J[i], db = propagate!(dw, w, b, X, Y)
     @. w = w - learning_rate * dw
        b = b - learning_rate * db
    end
    return J, w, b
end

"""
    predict(w, b, x)

Given model (w, b) and sample x, predict the class y.
"""
@inline function predict(w, b, x)
    a = sigmoid(dot(w, x) + b)
    return 1 * (a > 0.5)
end

end # module
