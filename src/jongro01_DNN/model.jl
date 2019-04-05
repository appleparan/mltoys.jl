using Random
using Printf

using BSON
using CSV
using CuArrays
using Distributions: sample
using StatsBase: mean_and_std

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad

ENV["MPLBACKEND"]="agg"

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.

function train_all(df, features, mb_idxs, output_size)
    opt = ADAM(0.01)

    # skip last segment becuase Y is a day after X finishes, so it could have bounds error
    
    PM10_batched_arr = [make_minibatch(df, "PM10", idx, feas, output_size) for idx in mb_idxs[1:end-1]]
    
    train_set = gpu.(PM10_batched_arr[train_idx])
    valid_set = gpu.(PM10_batched_arr[valid_idx])
    test_set = gpu.(PM10_batched_arr[test_idx])
    
    input_size = size(train_set[0][0])
    PM10_model, PM10_loss, PM10_accuracy =
        compile_PM10(df, feeatures, mb_idxs, train_set, valid_set, test_set)
    
    train(PM10_model, train_set, test_set, PM10_loss, PM10_accuracy, opt, 50, "/output/PM10.bson")
end

function train(model, train_set, test_set, loss, accuracy, opt, epoch, filename)

    @info("Beginning training loop...")
    
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:epoch
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Calculate accuracy:
        acc = accuracy(test_set...)
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
            cpu_model = cpu(model)
            BSON.@save filename cpu_model epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
end

function compile_PM10(dfs, features, input_size, output_size)
    @info("Constructing model...")
    # features
    
    model = Chain(
        Dense(input_size, 100),
        Dropout(0.2),

        Dense(100, 100),
        Dropout(0.2),

        Dense(100, 100),
        Dropout(0.2),

        Dense(100, output_size),
        Dropout(0.2),
        
        #NNlib.leakyrelu,
        softmax,
    )

    loss(x, y) = Flux.mse(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    model = gpu(model)

    model, loss, accuracy
end