using Random
using Printf

using BSON: @save, @load
using CSV
using CuArrays
using Distributions: sample
using ProgressMeter
using StatsBase: mean_and_std

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

ENV["MPLBACKEND"]="agg"

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.

function train_all(df::DataFrame, features::Array{Symbol}, mb_idxs::Array{Any},
    input_size::Integer, output_size::Integer, epoch_size::Integer,
    train_idx, valid_idx, test_idx)

    opt = ADAM(0.01)

    @info "PM10 Training..."
    flush(stdout)

    # free minibatch after training because of memory usage
    PM10_model = train(df, :PM10, features, mb_idxs,
    input_size, output_size, epoch_size, opt,
    train_idx, valid_idx, test_idx,  "/mnt/PM10.bson")
    
    nothing
end

function train(df::DataFrame, ycol::Symbol, features::Array{Symbol}, mb_idxs::Array{Any},
    input_size::Integer, output_size::Integer, epoch_size::Integer, opt,
    train_idx, valid_idx, test_idx, output_path::String)

    # construct symbol for compile
    compile = eval(Symbol(:compile, '_', ycol))

    model, loss, accuracy = compile(input_size, output_size)

    p = Progress(length(mb_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    batched_arr = [(ProgressMeter.next!(p); make_minibatch(df, ycol, collect(idx), features, output_size)) for idx in mb_idxs]
    # don't know why but `|> gpu` causes segfault in following lines 
    train_set = batched_arr[train_idx]
    valid_set = batched_arr[valid_idx]
    test_set = batched_arr[test_idx]
    
    train!(model, train_set, test_set, loss, accuracy, opt, epoch_size, output_path)

    @info "     Validation acc : ", accuracy(valid_set)
    flush(stdout)

    model
end

function train!(model, train_set, test_set, loss, accuracy, opt, epoch_size::Integer, filename::String)

    @info(" Beginning training loop...")
    flush(stdout)

    best_acc = 0.0
    last_improvement = 0
    acc = 0.0
    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Calculate accuracy:
        acc = accuracy(test_set)
        @info(@sprintf("epoch [%d]: Test accuracy: %.4f", epoch_idx, acc))
        flush(stdout)

        # If our accuracy is good enough, quit out.
        if acc < 0.01
            @info("     -> Early-exiting: We reached our target accuracy of 0.01")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info " -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_model = cpu(model)
            # TrackedReal cannot be writable, convert to Real
            @save filename cpu_model epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn("     -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            flush(stdout)

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn("     -> We're calling this converged.")
            flush(stdout)

            break
        end
    end
end

function compile_PM10(input_size::Integer, output_size::Integer)
    @info("     Compiling model...")

    model = Chain(
        Dense(input_size, 100, relu),
        Dropout(0.2),

        Dense(100, 100, relu),
        Dropout(0.2),

        Dense(100, 100, relu),
        Dropout(0.2),

        Dense(100, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x |> gpu), y)
    accuracy(data) = RSME(data, model)

    model, loss, accuracy
end

function RSME(dataset, model)
    # RSME
    acc = 0.0

    for (x, y) in dataset
        ŷ = model(x)
        @assert size(ŷ) == size(y)

        acc += sqrt(sum(abs2.(ŷ .- y)) / length(y))
    end

    # acc is incomplete TrackedReal, convert it to pure Real type
    Flux.Tracker.data(acc / length(dataset))
end