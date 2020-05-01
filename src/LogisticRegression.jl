module LogisticRegression

function sigmoid(z::Number)
    return 1.0 / (1.0 + exp(-z))
end

function sigmoid(z::AbstractVector)
    return sigmoid.(z)
end

end # module
