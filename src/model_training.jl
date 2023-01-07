"""
Functions related to the model training
"""

using ProgressBars
using Flux
using CUDA


"""

"""
function plot_learning_curves(train_loss::T, valid_loss::T) where T <: Vector

end


"""
    loss_metrics(model::Learner, loader, device)

Calculate the loss and accuracy for `model` using the data provided by `loader` on a given `device`
"""
function loss_metrics(learner::Learner, loader, device)
    loss = 0.0
    acc = 0.0
    m = 0
    for (x_batch, y_batch) in loader
        x, y = device(x_batch), device(y_batch)
        preds = learner.neural_network(x)
        loss += learner.loss(preds, y)
        acc += sum(Flux.onecold(preds) .== Flux.onecold(y))
        m += size(x)[end] # umber of samples in this batch
    end
    return loss / m, acc / m
end


"""
    train_model(learner::Learner, train_loader::T, valid_loader::T; epochs::Integer, device) where T <: DataLoader     

Train a model encapsulated by `learner` using `train_loading` and `valid_loader` for a total of `epochs`.
    Optinially, pass a device to train the model on the GPU with `device = gpu`.

https://juliamltutorials.github.io/image-classification/catsvsdogs/
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
"""
function train_model!(learner::Learner, train_loader::T, valid_loader::T; epochs::Integer, device) where T <: DataLoader

    train_losses, valid_losses = Vector{Float64}(), Vector{Float64}()

    model = device(learner.neural_network)
    
    for epoch in 1:epochs
        for (x_batch, y_batch) in ProgressBar(train_loader)
            x, y = device(x_batch), device(y_batch)
            gradients = Flux.gradient(() -> learner.loss(model(x), y), learner.trainable_params)
            Flux.Optimise.update!(learner.optimizer, learner.trainable_params, gradients)
            @show learner.loss(model(x), y)
            @show model(x)
            println("ae porra")
        end

        train_loss, train_acc = loss_metrics(model, train_loader, device)
        valid_loss, valid_acc = loss_metrics(model, valid_loader, device)

        @info "Epoch $epoch | Learning Rate: $(learner.optimizer.eta)\n" *
            "train loss \t$train_loss \tvalid loss \t$valid_loss\n" *
            "train acc \t$train_acc \tvalid acc \t$valid_acc\n"

#        push!(loss_train, epoch_train_loss)
#        push!(loss_valid, epoch_valid_loss)
    end
end
