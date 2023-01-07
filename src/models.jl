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
    optimizer = isnothing(optimizer) ? Flux.Optimise.Adam() : optimizer
    neural_network = Chain(
        Flux.flatten,
        Dense(prod(input_shape) => 500, relu),
        BatchNorm(500),
        Dense(500 => 400, relu),
        BatchNorm(400),
        Dense(400 => 300, relu),
        BatchNorm(300),
        Dense(300 => 100, relu),
        BatchNorm(100),
        Dense(100 => 1, sigmoid),
    )
    trainable_params = Flux.params(neural_network)
    return DenseNetwork(neural_network, loss, optimizer, input_shape, trainable_params)
end


mutable struct ConvNetwork <: Learner
    neural_network::Chain
    loss
    optimizer
    input_shape::Tuple{Int, Int, Int}  # (Height, Width, Channels)
    trainable_params
end


function ConvNetwork(; input_shape, loss, optimizer = nothing)

    optimizer = isnothing(optimizer) ? Flux.Optimise.Adam() : optimizer
    conv_output_shape = (div(input_shape[1], 2), div(input_shape[2],2), 32) 

    neural_network = Chain(
        Conv((3, 3), input_shape[end] => 32, relu; pad = SamePad()),
        MaxPool((2,2)),
        flatten,
        Dense(prod(conv_output_shape) => 128, relu),
        Dense(128 => 2),
        softmax,
    )

    trainable_params = Flux.params(neural_network)

    return DenseNetwork(neural_network, loss, optimizer, input_shape, trainable_params)
end
