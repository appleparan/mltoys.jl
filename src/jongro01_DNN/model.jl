using LinearAlgebra: norm
using Random
using Printf

using BSON: @save, @load
using CSV
using CuArrays
using Dates: now
using JuliaDB
using MicroLogging
using ProgressMeter
using StatsBase: mean_and_std

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

function train_all(df::DataFrame, norm_feas::Array{Symbol}, norm_prefix::String,
    input_size::Integer, batch_size::Integer, output_size::Integer, epoch_size::Integer,
    total_idxs::Array{Any,1}, train_chnk::Array{T,1}, valid_idxs::T, test_idxs::T,
    μσs::NDSparse) where T <: Array{Int64,1}

    @info "PM10 Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    PM10_model, PM10_μσ = train(df, :PM10, norm_prefix, norm_feas,
    input_size, batch_size, output_size, epoch_size,
    total_idxs, train_chnk, valid_idxs, test_idxs, μσs,
    "PM10")
    
    @info "PM25 Training..."
    flush(stdout); flush(stderr)

    PM25_model, PM25_μσ = train(df, :PM25, norm_prefix, norm_feas,
    input_size, batch_size, output_size, epoch_size,
    total_idxs, train_chnk, valid_idxs, test_idxs, μσs,
    "PM25")

    nothing
end

function train(df::DataFrame, ycol::Symbol, norm_prefix::String, norm_feas::Array{Symbol},
    input_size::Integer, batch_size::Integer, output_size::Integer, epoch_size::Integer,
    total_idxs::Array{Any,1}, train_chnk::Array{T,1}, valid_idxs::T, test_idxs::T, μσs::NDSparse, 
    filename::String) where T <: Array{Int64,1}

    norm_ycol = Symbol(norm_prefix, ycol)
    # extract from ndsparse
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value

    # compute mean and std by each train/valid/test set
    # merge chunks and get rows in df : https://discourse.julialang.org/t/very-best-way-to-concatenate-an-array-of-arrays/8672/17
    train_μ, train_σ = mean_and_std(df[reduce(vcat, train_chnk), norm_ycol])
    valid_μ, valid_σ = mean_and_std(df[valid_idxs, norm_ycol])
    test_μ, test_σ = mean_and_std(df[test_idxs, norm_ycol])
    μσ = ndsparse((
        dataset = ["train", "train", "valid", "valid", "test", "test"],
        type = ["μ", "σ", "μ", "σ", "μ", "σ"]),
        (value = [train_μ, train_σ, valid_μ, valid_σ, test_μ, test_σ],))

    # construct compile function symbol
    compile = eval(Symbol(:compile, '_', ycol))
    model, loss, accuracy, opt = compile(input_size, batch_size, output_size, μσ)

    # create (input(1D), output(1D)) pairs of total dataframe row
    @info "    Constructing (input, output) pairs..."
    p = Progress(length(total_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    input_pairs = [(ProgressMeter.next!(p); make_pairs(df, norm_ycol, collect(idx), norm_feas, input_size, output_size)) for idx in total_idxs]

    # construct minibatch for train_set
    # |> gpu doesn't work to *_set directly
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Constructing minibatch..."
    p = Progress(length(train_chnk), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    # total_idxs[chnk] = get list of pair indexes -> i.e. [1, 2, 3, 4]
    train_set = [(ProgressMeter.next!(p); make_minibatch(input_pairs, chnk, batch_size)) for chnk in train_chnk]

    # don't construct minibatch for valid & test sets
    valid_set = input_pairs[valid_idxs]
    test_set = input_pairs[test_idxs]

    train_set = train_set
    valid_set = valid_set
    test_set = test_set

    train!(model, train_set, test_set, loss, accuracy, opt, epoch_size, filename)

    # TODO : (current) validation with zscore, (future) validation with original valud?
    @info "    Validation acc : ", accuracy("valid", valid_set)
    flush(stdout); flush(stderr)
    #=
    /home/appleparan/.julia/packages/GR/Q8slp/src/../deps/gr/bin/gksqt: error while loading shared libraries: libQt5Widgets.so.5: cannot open shared object file: No such file or directory
    connect: Connection refused
    GKS: can't connect to GKS socket application
    Did you start 'gksqt'?
    =#
    plot_initdata(valid_set, ycol, total_μ, total_σ, "/mnt/")
    plot_DNN(valid_set, model, ycol, total_μ, total_σ, "/mnt/")
    # plot_DNN_toCSV(valid_set, model, total_μ, total_σ, png_path)

    model, μσ
end

function train!(model, train_set, test_set, loss, accuracy, opt, epoch_size::Integer, filename::String)

    @info("    Beginning training loop...")
    flush(stdout); flush(stderr)

    best_acc = 100.0
    last_improvement = 0
    acc = 0.0

    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Calculate accuracy:
        acc = accuracy("test", test_set)
        @info(@sprintf("epoch [%d]: Test accuracy: %.6f Time: %s", epoch_idx, acc, now()))
        flush(stdout); flush(stderr)

        # If our accuracy is good enough, quit out.
        if acc < 0.01
            @info("    -> Early-exiting: We reached our target accuracy (RSR) of 0.01")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc < best_acc
            @info "    -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_model = model |> cpu
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            @save filepath cpu_model epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 10 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn("    -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
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

function compile_PM10(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in PM10: ", unit_size
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y) + sum(norm, params(model))
    accuracy(setname, data) = RSR(setname, data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_PM25(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in PM25: ", unit_size
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y) + sum(norm, params(model))
    accuracy(setname, data) = RSR(setname, data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end
