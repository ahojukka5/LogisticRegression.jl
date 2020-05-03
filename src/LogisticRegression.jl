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
    evaluate_linear!(z, W, b, x)

Evaluate linear layer, i.e. calculate `z = W*x + b`.
"""
@inline function evaluate_linear!(
    z::AbstractVector,
    W::AbstractMatrix,
    b::AbstractVector,
    x::AbstractVector)
    mul!(z, W, x)
    z .+= b
    return z
end

"""
    propagate!(J, dw, db, W, b, X, Y, z, a, dz)

Forward- and backpropagation.

Given set of samples (X, Y), calculate total cost and partial derivatives with
respect to w and b. X and Y must be iterables with length of m, where m is the
number of samples. Algoritm writes to J, dw, db, z, a, and dz. The last three
auxiliary vectors for intemediate results and they size needs to equal b.
"""
function propagate!(
    J::AbstractVector,
    dW::AbstractMatrix,
    db::AbstractVector,
    W::AbstractMatrix,
    b::AbstractVector,
    X::AbstractVector{V},
    Y::AbstractVector,
    z::AbstractVector,
    a::AbstractVector,
    dz::AbstractVector
) where {V<:AbstractVector}
    @assert length(X) == length(Y)
    m = length(X)
    @assert m > 0
    fill!(J, 0.)
    fill!(dW, 0.0)
    fill!(db, 0.0)
    for (x, y) in zip(X, Y)
        z = evaluate_linear!(z, W, b, x)
     @. a = sigmoid(z)
     @. dz = a - y
     @. J += logistic_loss(y, a)
     @. dW += x' * dz
     @. db += dz
    end
 @. J /= m
 @. dW /= m
 @. db /= m
    return
end

"""
    optimize!(J, W, b, X, Y; num_iterations = 1000, learning_rate = 1.0)

Given initial model (J, W, b) and data (X, Y), optimize model parameters using
gradient descent method. Model optimization is done in place.
"""
function optimize!(
    J::AbstractVector,
    W::AbstractMatrix,
    b::AbstractVector,
    X::AbstractVector{T},
    Y::AbstractVector;
    num_iterations::Integer = 1000,
    learning_rate::Float64 = 1.0,
) where {T<:AbstractVector}
    dW = zero(W)
    db = zero(b)
    z = zero(b) 
    a = zero(b)
    dz = zero(b)
    for i in 1:num_iterations
        propagate!(J, dW, db, W, b, X, Y, z, a, dz)
     @. W = W - learning_rate * dW
     @. b = b - learning_rate * db
    end
    return J, W, b
end

"""
    predict(W, b, x)

Given model (W, b) and sample x, predict the class y.
"""
@inline function predict(W, b, x)
    return 1 * (sigmoid.(W*x .+ b) .> 0.5)
end

end # module
