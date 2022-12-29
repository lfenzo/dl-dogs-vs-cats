"""
Functions related to the model training
"""

using ProgressBars


"""

"""
function plot_learning_curves(train_loss::T, valid_loss::T) where T <: Vector

end

#function loss_metrics(model::Learner, loader, device)
#    acc = 0.0
#    m = 0
#    for (x, y) in loader
#        x, y = device(x_batch), device(y_batch)
#        acc += 
#    end
#end


"""
    
"""
function train_model(learner::Learner, train_loader::T, valid_loader::T; epochs::Integer, device) where T <: DataLoader

    train_loss, valid_loss = Vector{Float64}(), Vector{Float64}()
    
    for epoch in 1:epochs
        for (x_batch, y_batch) in ProgressBar(train_loader)
            x, y = device(x_batch), device(y_batch)
            gradients = Flux.gradient(() -> learner.loss(learner.neural_network(x), y), learner.trainable_params)
            Flux.Optimise.update!(optimizer, learner.trainable_params, gradients)
        end


#        epoch_train_loss = loss_metrics(model, train_loader, device)
#        epoch_valid_loss = loss_metrics(model, valid_loader, device)
#
#        @info "Epoch $epoch | Learning Rate: $(optimizer.eta)\n" *
#            "training loss \t$epoch_train_loss\n" *
#            "validation loss \t$epoch_valid_loss"
#
#        push!(loss_train, epoch_train_loss)
#        push!(loss_valid, epoch_valid_loss)
    end
end
