module LogisticRegression

function sigmoid(z::Number)
    return 1.0 / (1.0 + exp(-z))
end

end # module
