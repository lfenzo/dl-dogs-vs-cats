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


const IMAGE_DIR = joinpath("..", "img")


"""
    load_images(nsamples::Number) :: Tuple{Vectir, Vector}

Load `nsamples` images from the dataset directory. Note that `nsamples / 2` exemples from each
class are returned from this function, not `nsamples`.
"""
function load_images(; nsamples::Int = 2000, input_shape::Tuple = (128, 128))

    # problematic images that must be skipped
    # (only obtained during the executions of the data loading)
    # For now, we are not covering the case in which the random selected samples happen to be one
    # of the problematic ones
    problematic_images = ["666.jpg", "11702.jpg", "1789.jpg"] 

    # could as well be "cats" since both sub-dirs have the same number of examples
    all_image_indexes = 1:length(readdir(joinpath(IMAGE_DIR, "dogs")))
    random_image_indexes = StatsBase.sample(all_image_indexes, trunc(Int32, nsamples / 2), replace = false)

    cats_vector = Vector()
    dogs_vector = Vector()

    for (container, dataset) in zip([cats_vector, dogs_vector], ["cats", "dogs"])
        for image in readdir(joinpath("..", "img", dataset))[random_image_indexes]
            if !(image in problematic_images)
                
                # uses FileIO to read the image files to an image object
                raw_image = FileIO.load(joinpath("../img/$dataset", image))
                resized = imresize(raw_image, input_shape)
                
                # converts the image object into an H x W x C array (we are using (H,W,C,B)-shaped data)
                # channelview converts the image from a matrix with RGB elements to a tensor with
                # (height * width * chanels) format
                image_array = Array{Float32}(permutedims(channelview(resized), (2, 3, 1)))

                push!(container, image_array)
            end
        end
    end

    dogs = reduce((a, b) -> cat(a, b, dims = 4), dogs_vector)
    cats = reduce((a, b) -> cat(a, b, dims = 4), cats_vector)

    return dogs, cats
end


"""
    prepare_data(positive_class, negative_class; fractions::Tuple, batchsize::Integer) :: Tuple

Prepare the data contained in the `positive_class` and `negative_class` containers into train, valid
and test dataloaders with batches with `batchsizse` samples.
"""
function prepare_data(positive_class, negative_class; fractions::Tuple, batchsize::Integer)

    # array in the format (128, 128, 3, size)
    images_tensor = cat(positive_class, negative_class, dims = 4)

    # since both class containers have the same number of samples, we could have used the negative
    # classe here as well
    nsamples = size(positive_class, 4) # the 4th dimention gives us the number of examples

    labels = vcat(repeat([1], nsamples), repeat([0], nsamples)) 
    onehot_labels = Flux.onehotbatch(labels, [0, 1], 1)

    return [
        DataLoader(dataset, shuffle = true, batchsize = batchsize)
        for dataset in splitobs((images_tensor, onehot_labels), at = fractions)
    ]
end
