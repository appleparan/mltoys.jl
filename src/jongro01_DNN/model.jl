using Random
using Printf

using BSON
using CSV
using CuArrays
using Distributions: sample
using DataStructures: CircularBuffer

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad

ENV["MPLBACKEND"]="agg"
function get_input(input_path)
    df = CSV.read(input_path)
    @show first(df, 5)
    @show size(df)
    df
end

function split_size(df)
    tot_size = size(df)[1]
    
    # train : valid : test = 0.64 : 0.16 : 0.20
    train_size = round(tot_size * 0.64)
    valid_size = round(tot_size * 0.16)
    test_size = tot_size - (train_size + valid_size)

    tot_size, Int(train_size), Int(valid_size), Int(test_size)
end

function perm_idx(tot_size, train_size, valid_size, test_size)
    tot_idx = collect(range(1, stop=tot_size))
    
    tot_idx = Random.randperm(tot_size)
    train_idx = tot_idx[1: train_size]
    valid_idx = tot_idx[train_size + 1: train_size + valid_size]
    test_idx = tot_idx[train_size + valid_size + 1: end]

    sort!(train_idx), sort!(valid_idx), sort!(test_idx)
end

function batch(arr, s)
    batches = []
    l = size(arr, 1)
    for i=1:s:l
        push!(batches, arr[i:min(i+s-1, l), :])
    end
    batches
end

function perm_df(df, permed_idx, col, labels)
    X = df[permed_idx, labels]
    Y = df[permed_idx, col]

    X, Y
end

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.

function train_all()
    df = get_input("/home/appleparan/input/jongro_single.csv")
   
    batch_size = 72
    total_size, train_size, valid_size, test_size = split_size(df)
    train_idx, valid_idx, test_idx = perm_idx(total_size, train_size, valid_size, test_size)
    # 
    
    opt = ADAM(0.01)

    NO2_train, NO2_valid, NO2_test, NO2_model, NO2_loss, NO2_accuracy =
        prepare_train_NO2(df, total_size, batch_size, train_idx, valid_idx, test_idx)
    
    train(NO2_model, NO2_train, NO2_test, NO2_loss, NO2_accuracy, opt, 50, "NO2.bson")
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

function prepare_train_NO2(df, total_size, batch_size, train_idx, valid_idx, test_idx)
    @info("Constructing model...")
    # features
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid]
    model = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((batch_size,), len(features)=>100),
        Dropout(0.2),

        Dense(100, 100),
        Dropout(0.2),

        Dense(100, 100),
        Dropout(0.2),

        Dense(100, 24),
        Dropout(0.2),

        
        #NNlib.leakyrelu,
        softmax,
    )

    loss(x, y) = Flux.mse(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    # train set
    NO2_train = perm_df(df, :NO2, features, train_idx)
    NO2_valid = perm_df(df, :NO2, features, valid_idx)
    NO2_test = perm_df(df, :NO2, features, test_idx)

    NO2_train = gpu.(NO2_train)
    NO2_valid = gpu.(NO2_valid)
    NO2_test = gpu.(NO2_test)
    model = gpu(model)

    NO2_train, NO2_valid, NO2_test, model, loss, accuracy
end