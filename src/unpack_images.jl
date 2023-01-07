using MLJ

const IMAGE_DIR = joinpath("..", "img")

"""

Unpack and stratify all images into the img/ directory in the following structure:
    img/
        train/ (70% of images)
        valid/ (15% of images)
        test/  (15% of images)
"""
function main()

    # unzup the zipped directory to the src/ dir (they will be later deleted from there)
    run(Cmd(`unzip dogs-vs-cats.zip`, dir = ".."))
    run(Cmd(`unzip train.zip`, dir = ".."))
    mv("../train", "../img", force = true)

    # remove unnecessary files
    for artifact in ["sampleSubmission.csv", "test1.zip", "train.zip"]
        rm(joinpath("..", artifact))
    end

    all_images = readdir(IMAGE_DIR)
    dogs = filter(s -> occursin("dog", s), all_images)
    cats = filter(s -> occursin("cat", s), all_images)

    images = []

    # since we have the same number of samples in bath classes we can interleave the observations
    # here in order to have the splits automatically stratified
    for i in 1:length(cats)
        push!(images, dogs[i])
        push!(images, cats[i])
    end

    train, valid, test = partition(images, 0.7, 0.15)

    for (dataset, dataset_label) in zip([train, valid, test], ["train", "valid", "test"])
        !isdir(joinpath(IMAGE_DIR, dataset)) && mkdir(joinpath(IMAGE_DIR, dataset))
        for img in dataset
            try
                run(`cp $(IMAGE_DIR)/$(img) $(IMAGE_DIR)/$(dataset_label)/$(img)`)
                run(`rm $(IMAGE_DIR)/$(img)`)
            catch e
                if isa(e, LoadError)
                    println("skipping $img")
                    continue
                end
            end
        end
    end
end


main()
