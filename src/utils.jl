import MLUtils: getobs, numobs

struct ImageDataset
    files::Vector{String}
end

numobs(data::ImageDataset) = length(data.files)

function ImageDataset(dir::String)
    # problematic images that must be removed from the directory
    problematic_images = ["666.jpg", "11702.jpg", "1789.jpg", "6245.jpg", "9078.jpg"] 
    # removing the problematic images from the image directory
    for img in readdir(dir)
        img in problematic_images && rm(joinpath(IMAGE_DIR, img))
    end
    return ImageDataset(readdir(dir, join = true))
end

function getobs(data::ImageDataset, idx::Vector{Int})

    batch_images = []
    batch_labels = Vector{Int}()

    for image in data.files[idx]
        raw_image = FileIO.load(image)
        resized = imresize(raw_image, IMAGE_SHAPE)  # resizeing the image to an standark size
        channel_view = Array{Float32}(channelview(resized))[1:3, :, :, 1]  # handling images with 4 dimensions (not channels)

        push!(batch_images, permutedims(channel_view, (2, 3, 1)))
        push!(batch_labels, occursin("dog", image) ? 1 : 0)  # assigning 1 for "dog" and 0 for "cats"
    end

    onehot_labels = convert.(Float32, Flux.onehotbatch(batch_labels, [1, 0]))

    return cat(batch_images..., dims = 4), onehot_labels
end
