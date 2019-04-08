using Random
using Printf

using BSON: @save, @load
using CSV
using CuArrays
using Dates: now
using IndexedTables
using MicroLogging
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

    PM10_train_μ, PM10_train_σ = mean_and_std(df[train_idx, :PM10])
    PM10_valid_μ, PM10_valid_σ = mean_and_std(df[valid_idx, :PM10])
    PM10_test_μ, PM10_test_σ = mean_and_std(df[test_idx, :PM10])
    PM10_μσ = ndsparse((
        dataset = ["train", "train", "valid", "valid", "test", "test"],
        type = ["μ", "σ", "μ", "σ", "μ", "σ"]),
        (value = [PM10_train_μ, PM10_train_σ, PM10_valid_μ, PM10_valid_σ, PM10_test_μ, PM10_test_σ],))

    PM25_train_μ, PM25_train_σ = mean_and_std(df[train_idx, :PM25])
    PM25_valid_μ, PM25_valid_σ = mean_and_std(df[valid_idx, :PM25])
    PM25_test_μ, PM25_test_σ = mean_and_std(df[test_idx, :PM25])
    PM25_μσ = ndsparse((
        dataset = ["train", "train", "valid", "valid", "test", "test"],
        type = ["μ", "σ", "μ", "σ", "μ", "σ"]),
        (value = [PM25_train_μ, PM25_train_σ, PM25_valid_μ, PM25_valid_σ, PM25_test_μ, PM25_test_σ],))

    @info "PM10 Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    PM10_model = train(df, :PM10, features, mb_idxs,
    input_size, output_size, epoch_size, opt,
    train_idx, valid_idx, test_idx, PM10_μσ, "/mnt/PM10.bson")
    
    @info "PM25 Training..."
    flush(stdout); flush(stderr)

    PM25_model = train(df, :PM25, features, mb_idxs,
    input_size, output_size, epoch_size, opt,
    train_idx, valid_idx, test_idx, PM25_μσ, "/mnt/PM25.bson")

    nothing
end

function train(df::DataFrame, ycol::Symbol, features::Array{Symbol}, mb_idxs::Array{Any},
    input_size::Integer, output_size::Integer, epoch_size::Integer, opt,
    train_idx, valid_idx, test_idx, μσ, output_path::String)

    # construct symbol for compile
    compile = eval(Symbol(:compile, '_', ycol))

    model, loss, accuracy = compile(input_size, output_size, μσ)

    p = Progress(length(mb_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    batched_arr = [(ProgressMeter.next!(p); make_minibatch(df, ycol, collect(idx), features, output_size)) for idx in mb_idxs]
    # don't know why but `|> gpu` causes segfault in following lines 
    train_set = batched_arr[train_idx]
    valid_set = batched_arr[valid_idx]
    test_set = batched_arr[test_idx]
    
    train!(model, train_set, test_set, loss, accuracy, opt, epoch_size, output_path)

    @info "     Validation acc : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    model
end

function train!(model, train_set, test_set, loss, accuracy, opt, epoch_size::Integer, filename::String)

    @info(" Beginning training loop...")
    flush(stdout); flush(stderr)

    best_acc = 100.0
    last_improvement = 0
    acc = 0.0
    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Calculate accuracy:
        acc = accuracy(test_set)
        @info(@sprintf("epoch [%d]: Test accuracy: %.6f Time: %s", epoch_idx, acc, now()))
        flush(stdout); flush(stderr)

        # If our accuracy is good enough, quit out.
        if acc < 0.5
            @info("     -> Early-exiting: We reached our target accuracy (RSR) of 0.5")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc <= best_acc
            @info " -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_model = cpu(model)
            # TrackedReal cannot be writable, convert to Real
            @save filename cpu_model epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 10 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn("     -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            flush(stdout); flush(stderr)

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 20
            @warn("     -> We're calling this converged.")
            flush(stdout); flush(stderr)

            break
        end
    end
end

function compile_PM10(input_size::Integer, output_size::Integer, μσ)
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
    accuracy(data) = RSR(data, model, μσ["train", "μ"][:value])

    model, loss, accuracy
end

function compile_PM25(input_size::Integer, output_size::Integer, μσ)
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
    accuracy(data) = RSR(data, model, μσ["train", "μ"][:value])

    model, loss, accuracy
end
