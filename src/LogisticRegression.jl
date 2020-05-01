module LogisticRegression

function sigmoid(z::Number)
    return 1.0 / (1.0 + exp(-z))
end

function sigmoid(z::Vector)
    return sigmoid.(z)
end

end # module
