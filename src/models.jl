using Flux
using Metalhead


abstract type Learner end

mutable struct DenseNetwork <: Learner
    neural_network::Chain
    loss
    optimizer
    input_shape::Tuple{Int, Int, Int}  # (Height, Width, Channels)
    trainable_params
end

function DenseNetwork(; input_shape, loss, optimizer = nothing)
    println(input_shape)
    optimizer = isnothing(optimizer) ? Flux.Optimise.Adam() : optimizer
    neural_network = Chain(
        Flux.flatten,
        Dense(reduce(*, input_shape) => 500, relu),
        BatchNorm(500),
        Dense(500 => 400, relu),
        BatchNorm(400),
        Dense(400 => 300, relu),
        BatchNorm(300),
        Dense(300 => 100, relu),
        BatchNorm(100),
        Dense(100 => 2),
        softmax,
    )
    trainable_params = Flux.params(neural_network)
    return DenseNetwork(neural_network, loss, optimizer, input_shape, trainable_params)
end


