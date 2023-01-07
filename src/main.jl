using MLJ
using Flux
using FileIO
using Random
using Images
using MLUtils
using ImageView
using Augmentor
using StatsBase
using ProgressBars
using BenchmarkTools

const IMAGE_DIR = joinpath("..", "img")
const IMAGE_SHAPE = (200, 200)

include("utils.jl")
include("models.jl")
include("model_training.jl")


function main()
    epochs = 20
    batchsize = 8

    dataloaders = Dict(
        loader => DataLoader(ImageDataset(joinpath(IMAGE_DIR, loader));
            batchsize = batchsize,
            shuffle = true,
        )
        for loader in ["train", "valid", "test"]
    )

    model = ConvNetwork(
        input_shape = (IMAGE_SHAPE..., 3),  # adding the extra channels dimension
        loss = Flux.Losses.crossentropy,
        optimizer = Flux.Adam(0.0001),
    )

    trained_model, train_loss, valid_loss = train_model!(
        model, dataloaders["train"], dataloaders["valid"];
        epochs = epochs,
        device = gpu,
    )
end


main()
