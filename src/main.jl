include("utils.jl")
include("models.jl")
include("model_training.jl")


const DATA_DIR = joinpath("..", "data")

function main()

    input_shape = (128, 128)
    epochs = 20
    batchsize = 64

    # if this directory is empty, then generate the data from the img/ directory and store it in data/
    # (meant to be executed only once by project instantiation)
    if isempty(readdir(DATA_DIR))
        @info "Loading images, this will take a while..."
        dogs, cats = image_batch_to_tensor("../img", input_shape = input_shape) 
        FileIO.save(joinpath(DATA_DIR, "image_arrays.jld2"), Dict("dogs" => dogs, "cats" => cats))
    else
        dogs, cats = FileIO.load(joinpath(DATA_DIR, "image_arrays.jld2"), "dogs", "cats")
    end

    train_loader, valid_loader, test_loader = prepare_data(dogs, cats, fractions = (0.7, 0.15), batchsize = batchsize)

    model = DenseNetwork(
        input_shape = (input_shape..., 3),  # adding the extra channels dimension
        loss = Flux.Losses.binarycrossentropy,
        optimizer = Flux.NAdam(),
    )

    trained_model, train_loss, valid_loss = train_model(
        model, train_loader, valid_loader;
        epochs = epochs,
        device = gpu,
    )
end


main()
