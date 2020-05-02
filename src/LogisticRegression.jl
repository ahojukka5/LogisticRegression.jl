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

"""
    optimize!(w, b, X, Y, num_iterations, learning_rate)

Given initial model (w, b) and data (X, Y), optimize model parameters using
gradient descent method. Model optimization is done in place. Returns optimized
parameters and cost for each iteration for convergence studies.
"""
function optimize!(
    w::AbstractVector,
    b::Number,
    X::AbstractVector{T},
    Y::AbstractVector,
    num_iterations::Integer,
    learning_rate::Float64,
) where {T<:AbstractVector}
    J = zeros(num_iterations)
    for i in 1:num_iterations
        result = propagate(w, b, X, Y)
        J[i] = result.J
        w = w - learning_rate * result.dw
        b = b - learning_rate * result.db
    end
    return w, b, J
end

"""
    optimize_gd!(w, b, X, Y, num_iterations, learning_rate)

Given initial model (w, b) and data (X, Y), optimize model parameters using
gradient descent method. Model optimization is done in place. Returns optimized
parameters and cost for each iteration for convergence studies. This is the same
algorithm than above, only slightly optimized with respect to memory useage and
speed.
"""
function optimize_gd!(
    w::AbstractVector,
    b::Number,
    X::AbstractVector{T},
    Y::AbstractVector,
    num_iterations::Integer,
    learning_rate::Float64,
) where {T<:AbstractVector}
    @assert length(X) == length(Y)
    m = length(X)
    @assert m > 0

    costs = zeros(num_iterations)
    dw = zeros(length(w))
    for k in 1:num_iterations
        fill!(dw, 0.0)
        J = 0.0
        db = 0.0
        for (x, y) in zip(X, Y)
            z = dot(w, x) + b
            a = sigmoid(z)
            J += -1.0 * (y * log(a) + (1.0 - y) * log(1.0 - a))
            dz = a - y
            @. dw = dw + x * dz
            db += dz
        end
        costs[k] = J
        @. w = w - learning_rate * dw / m
        b = b - learning_rate * db / m
    end
    return w, b, costs
end

end # module
