using MLJ
using Flux
using FileIO
using Random
using Images
using Logging
using MLUtils
using ImageView
using Augmentor
using StatsBase
using ProgressBars
using BenchmarkTools

function build_image_batches(images::Vector{String}; batchsize::Int = 500) :: Vector{Vector{String}}
    image_batches = []
    for i in 1:batchsize:length(images)
        push!(image_batches, images[i:i + batchsize - 1])
    end
    return image_batches
end


"""
    image_batch_to_tensor(path; input_shape::Tuple = (128, 128))

Load `nsamples` images from the dataset directory. Note that `nsamples / 2` exemples from each
class are returned from this function, not `nsamples`.

This function is meant to be run only once per project instantiation bacause it is rather slow
"""
function image_batch_to_tensor(path; input_shape::Tuple = (128, 128))

    Logging.disable_logging(Logging.Warn)

    # problematic images that must be skipped
    # (only obtained during the executions of the data loading)
    # For now, we are not covering the case in which the random selected samples happen to be one
    # of the problematic ones
    problematic_images = ["666.jpg", "11702.jpg", "1789.jpg", "6245.jpg", "9078.jpg"] 

    class_tensors = []

    for class in ["dogs", "cats"]
        concatenated_tensors = []
        for image_batch in build_image_batches(readdir(joinpath("..", "img", class)), batchsize = 500)
            image_array_container = Vector()
            for image in image_batch
                if !(image in problematic_images)
                    
                    # uses FileIO to read the image files to an image object
                    raw_image = FileIO.load(joinpath("../img/$class", image))
                    resized = imresize(raw_image, input_shape)

                    # handling images with 4 dimensions (not channels)
                    channel_view = Array{Float32}(channelview(resized))[1:3, :, :, 1]

                    # permuting the dimenrions of the image so that we have images with (H,W,C) format
                    image_array = permutedims(channel_view, (2, 3, 1))

                    # some images have only one channel, so given the nubmber of available images,
                    # we can skip them in this part
                    if size(image_array) == (128, 128, 3)
                        push!(image_array_container, image_array)
                    end
                end
            end
           push!(concatenated_tensors, cat(image_array_container..., dims = 4))
        end
        push!(class_tensors, cat(concatenated_tensors..., dims = 4))
    end

    return [class_tensors...]
end


"""
    prepare_data(positive_class, negative_class; fractions::Tuple, batchsize::Integer) :: Tuple

Prepare the data contained in the `positive_class` and `negative_class` containers into train, valid
and test dataloaders with batches with `batchsizse` samples.
"""
function prepare_data(positive_class, negative_class; fractions::Tuple, batchsize::Integer)

    # array in the format (128, 128, 3, size)
    @time images_tensor = cat(positive_class, negative_class, dims = 4)
    println(size(images_tensor))

    # the 4th dimention gives us the number of examples
    nsamples_positive = size(positive_class, 4)
    nsamples_negative = size(negative_class, 4) 

    labels = vcat(repeat([1], nsamples_positive), repeat([0], nsamples_negative)) 
    onehot_labels = Flux.onehotbatch(labels, [0, 1], 1)
    println(size(onehot_labels))

    return [
        DataLoader(dataset, shuffle = true, batchsize = batchsize)
        for dataset in splitobs((images_tensor, onehot_labels), at = fractions)
    ]
end
