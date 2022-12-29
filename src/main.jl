include("utils.jl")
include("models.jl")
include("model_training.jl")


function main()
    input_shape = (128, 128)
    epochs = 3
    batchsize = 64

#    input_shape = (input_shape..., 3)
#    return reduce(*, input_shape) 

    dogs, cats = load_images(nsamples = 128, input_shape = input_shape)

    model = DenseNetwork(
        input_shape = (input_shape..., 3),  # adding the extra channels dimension
        loss = Flux.Losses.binarycrossentropy,
        optimizer = Flux.NAdam(),
    )

    train_loader, valid_loader, test_loader = prepare_data(dogs, cats, fractions = (0.7, 0.15), batchsize = batchsize)

    #trained_model, train_loss, valid_loss = train_model(
    return train_model(
        model, train_loader, valid_loader;
        epochs = epochs,
        device = cpu,
    )
end


main()
