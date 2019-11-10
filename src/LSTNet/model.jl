"""
    train(df, ycol, norm_prefix, norm_feas,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        μσs, filename, test_dates)

"""
function train_LSTNet(train_wd::Array{DataFrame, 1}, valid_wd::Array{DataFrame, 1}, test_wd::Array{DataFrame, 1}, 
    ycol::Symbol, norm_prefix::String, feas::Array{Symbol},
    train_size::Integer, valid_size::Integer, test_size::Integer,
    sample_size::Integer, batch_size::Integer, kernel_length::Integer,
    output_size::Integer, epoch_size::Integer, eltype::DataType,
    μσs::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where I <: Integer

    @info "LSTNet training starts.."

    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in feas]

    # extract from ndsparse
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value
    total_min = float(μσs[String(ycol), "minimum"].value)
    total_max = float(μσs[String(ycol), "maximum"].value)

    # compute mean and std by each train/valid/test set
    μσ = ndsparse((
        dataset = ["total", "total"],
        type = ["μ", "σ"]),
        (value = [total_μ, total_σ],))

    # construct compile function symbol
    compile = eval(Symbol(:compile, "_", ycol, "_LSTNet"))
    model, state, loss, accuracy, opt = 
        compile((kernel_length, length(feas)), batch_size, output_size)

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Construct Training Set batch..."
    p = Progress(div(length(train_wd), batch_size) + 1, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_feas,
            sample_size, kernel_length, output_size, batch_size, eltype))
        for dfs in Base.Iterators.partition(train_wd, batch_size)]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet([df], norm_ycol, norm_feas,
            sample_size, kernel_length, output_size, 1, eltype))
        for df in valid_wd]

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    # batch_size should be 1 except raining
    test_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet([df], norm_ycol, norm_feas,
            sample_size, kernel_length, output_size, 1, eltype))
        for df in test_wd]

    # *_set : normalized
    df_evals = train_LSTNet!(state, model, train_set, valid_set, loss, accuracy, opt, epoch_size, μσ, norm_feas, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    eval_metrics = [:RMSE, :MAE, :MSPE, :MAPE]
    
    # test set evaluation
    for metric in eval_metrics
        metric_func = :($(metric)(test_set, model, μσ))
        @info " $(string(ycol)) $(string(metric)) for test   : ", eval(metric_func)
    end

    # valid set evaluation
    for metric in eval_metrics
        metric_func = :($(metric)(valid_set, model, μσ))
        @info " $(string(ycol)) $(string(metric)) for valid  : ", eval(metric_func)
    end

    if ycol == :PM10 || ycol == :PM25
        forecast_all, forecast_high = classification(test_set, model, ycol, μσ)
        @info " $(string(ycol)) Forecasting accuracy (all) for test : ", forecast_all
        @info " $(string(ycol)) Forecasting accuracy (high) for test : ", forecast_high
    end
    if ycol == :PM10 || ycol == :PM25
        forecast_all, forecast_high = classification(test_set, model, ycol, μσ)
        @info " $(string(ycol)) Forecasting accuracy (all) for test : ", forecast_all
        @info " $(string(ycol)) Forecasting accuracy (high) for test : ", forecast_high
    end

    # create directory per each time
    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath("/mnt/$(i_pad)/")
    end

    #dnn_table = predict_model_norm(test_set, model, ycol, total_μ, total_σ, output_size, "/mnt/")
    dnn_table = predict_model_minmax(test_set, model, ycol, total_min, total_max, output_size, "/mnt/")
    dfs_out = export_CSV(DateTime.(test_dates), dnn_table, ycol, output_size, "/mnt/", String(ycol))
    df_corr = compute_corr(dnn_table, output_size, "/mnt/", String(ycol))

    plot_DNN_scatter(dnn_table, ycol, output_size, "/mnt/", String(ycol))
    plot_DNN_histogram(dnn_table, ycol, output_size, "/mnt/", String(ycol))

    plot_datefmt = @dateformat_str "yyyymmddHH"

    plot_DNN_lineplot(DateTime.(test_dates), dnn_table, ycol, output_size, "/mnt/", String(ycol))
    plot_corr(df_corr, output_size, "/mnt/", String(ycol))

    # 3 months plot
    # TODO : how to generalize date range? how to split based on test_dates?
    # 1/4 : because train size is 3 days, result should be start from 1/4
    # 12/29 : same reason 1/4, but this results ends with 12/31 00:00 ~ 12/31 23:00
    
    push!(eval_metrics, :learn_rate)
    plot_evaluation(df_evals, ycol, eval_metrics, "/mnt/")

    model, μσ
end

function train_LSTNet!(state, model, train_set, valid_set, loss, accuracy, opt,
    epoch_size::Integer, μσ::AbstractNDSparse, norm_feas::Array{Symbol},
    filename::String) where {F <: AbstractFloat, T2 <: Tuple{AbstractArray{F, 2}, AbstractArray{F, 2}},
    T1 <: Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}}}

    @info("    Beginning training loop...")
    flush(stdout); flush(stderr)

    _acc = 0.0
    # for adjR2 -∞ is the worst
    best_acc = -Inf
    last_improvement = 0

    df_eval = DataFrame(epoch = Int64[], learn_rate = Float64[], ACC = Float64[],
        RMSE = Float64[], MAE = Float64[], MSPE = Float64[], MAPE = Float64[])

    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        @show size(train_set[1][1])
        @show size(train_set[1][2])
        # train model with normalized data set
        Flux.train!(loss, Flux.params(state), train_set, opt)

        # record evaluation
        rmse, mae, mspe, mape =
            evaluations(valid_set, model, μσ, norm_feas,
            [:RMSE, :MAE, :MSPE, :MAPE])
        push!(df_eval, [epoch_idx opt.eta _acc rmse mae mspe mape])

        # Calculate accuracy:
        _acc = Tracker.data(accuracy(valid_set))
        _loss = Tracker.data(loss(train_set[1][1], train_set[1][2]))
        @info(@sprintf("epoch [%d]: loss[1]: %.8E Valid accuracy: %.8f Time: %s", epoch_idx, _loss, _acc, now()))
        flush(stdout); flush(stderr)

        # If our accuracy is good enough, quit out.
        if _acc > 0.999
            @info("    -> Early-exiting: We reached our target accuracy")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if _acc > best_acc
            @info "    -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_state = state |> cpu
            weights = Tracker.data.(Flux.params(cpu_state))
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            μ, σ = μσ["total", "μ"].value, μσ["total", "σ"].value
            BSON.@save filepath cpu_state weights epoch_idx _acc μ σ
            best_acc = _acc
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

    df_eval
end

"""
    compile_PM10_LSTNetRecurSkip(kernel_size, sample_size, output_size)

kernel_size is 2D tuple for CNN kernel
sample_size and output_size is Integer
"""
function compile_PM10_LSTNet(
    kernel_size::Tuple{I, I},
    sample_size::I, output_size::I) where I<:Integer

    @info("    Compiling model...")
    # hidCNN ; # of output channel 
    hidCNN = 16
    # hidRNn : length of context vector
    hidRNN = 24

    kernel_length = kernel_size[1]
    # WHCN order : reverse to python framework (row-major && column-major)
    # W : Width = # of variables (= # of features)
    # H : Height = # of timesteps
    # C : Channel = 1
    # N : batches

    # 1. CNN
    #   extract short-term patterns in the time dimension 
    #   as well as local dependencies between variables
    #   * Channel : 1 => hidCNN 
    #   * Kernel Size : (kernel_length, length(features)
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidCNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidCNN, leakyrelu),
        Dropout(0.2)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidCNN, batch_size),...]
    # modelGRU is a Single GRU Cell
    # hidCNN is a embedding size in NLP
    # sample_size is a sequnce length in NLP
    # Output has same dimension as input
    #
    # Check figure in 'Build the Model' Section in 
    # https://www.tensorflow.org/tutorials/text/text_generation
    # SEQ_LENGTH = sample_size
    # Char Embdding = hidCNN
    # logits = output value itself
    modelTrainer = GRU(hidCNN, hidRNN) |> gpu
    modelPredictor = GRU(hidRNN, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Chain(
        Dense(hidRNN, 1),
        Dropout(0.2)) |> gpu

    # predict output seqeunce recursively by current state yhat
    # this starts after update state after input sequence
    # yhat must be single array (hidRNN, batch_size)
    # using Channel, implement Lazy sequence for recursive predict_recur and collect them.
    # check fib_ch
    function predict_recur(_yhat, seq_len::Integer)
        yhat = _yhat
        # (seq_len x batch_size) array for output
        c = []

        @show seq_len
        @show size(_yhat)
        for i in 1:seq_len
            # predict next output by giving input as one by one
            yhat2 = modelPredictor.(yhat)
            push!(c, modelDNN(yhat2))
            yhat = yhat2
        end

        @show size(c)
        @show size(c[1])
        @show size(transpose(hcat(c...)))
        # return as (seq_len x batch_size )
        #if ndims(_yhat) == 1
            return transpose(hcat(c...))
        #else 
        #    return c
        #end
    end

    state = (modelCNN, modelTrainer, modelPredictor, modelDNN)

    # define model by combining multiple models
    # x : batched input for CNN (sample_size, (length(features), 1, batch_size))
    function model(x)
        # to avoid segfault due to GC
        GC.@preserve modelCNN modelTrainer modelPredictor modelDNN

        # do CNN to extract local features
        ŷ_CNN = modelCNN(x)

        # 4D -> Array of 2Ds (hidCNN, batch)
        # [(hidCNN, batch), ...]
        Flux.reset!(modelTrainer)
        Flux.reset!(modelPredictor)
        x_RNN = unpack_seq(ŷ_CNN)

        # RNN
        # `modelGRU.` performs GRU for input sequence
        # `predict_GRU` performs prediction after input seqeunce
        y_RNN_end = modelTrainer.(x_RNN)

        # copy hidden state from Trainer
        modelPredictor.state = modelTrainer.state

        # y_RNN_end[end] is a last sequence,
        # collect
        ŷ_RNN = predict_recur(y_RNN_end[end], output_size)
        #Flux.reset!(state)

        ŷ_RNN
    end

    # Each row of model(x) is different result from model
    # Not sure to use loss function as matrix, so just mse. to each array and norm whole result
    #loss(x, y) = Flux.mse(model(x), y) + sum(LinearAlgebra.norm, Flux.params(model))
    #loss(x, y) = LinearAlgebra.norm(Flux.mse.(model(x), matrix2arrays(y)))
    # not to use LinearAlgebra.norm because of force promotion to Float64
    # which is not compatible to TrackedReal
    loss(x, y) = sqrt(sum((Flux.mse.(model(x), matrix2arrays(y))).^2))
    
    # TODO : How to pass feature size
    accuracy(data) = RMSE(data, model, μσ)
    opt = Flux.ADAM()

    model, state, loss, accuracy, opt
end

function compile_PM25_LSTNet(
    kernel_size::Tuple{I, I},
    sample_size::I, output_size::I) where I<:Integer

    @info("    Compiling model...")
    # hidCNN ; # of output channel 
    hidCNN = 24
    # hidRNn : length of context vector
    hidRNN = 24

    kernel_length = kernel_size[1]
    # WHCN order : reverse to python framework (row-major && column-major)
    # W : Width = # of variables (= # of features)
    # H : Height = # of timesteps
    # C : Channel = 1
    # N : batches

    # 1. CNN
    #   extract short-term patterns in the time dimension 
    #   as well as local dependencies between variables
    #   * Channel : 1 => hidCNN 
    #   * Kernel Size : (kernel_length, length(features)
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidCNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidCNN, leakyrelu),
        Dropout(0.2)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidCNN, batch_size),...]
    # modelGRU is a Single GRU Cell
    # hidCNN is a embedding size in NLP
    # sample_size is a sequnce length in NLP
    # Output has same dimension as input
    #
    # Check figure in 'Build the Model' Section in 
    # https://www.tensorflow.org/tutorials/text/text_generation
    # SEQ_LENGTH = sample_size
    # Char Embdding = hidCNN
    # logits = output value itself
    modelTrainer = GRU(hidCNN, hidRNN) |> gpu
    modelPredictor = GRU(hidRNN, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Chain(
        Dense(hidRNN, 1),
        Dropout(0.2)) |> gpu

    # predict output seqeunce recursively by current state yhat
    # this starts after update state after input sequence
    # yhat must be single array (hidRNN, batch_size)
    # using Channel, implement Lazy sequence for recursive predict_recur and collect them.
    # check fib_ch
    function predict_recur(yhat, out_seq_len::Integer)
        _yhat = yhat
        # (seq_len x batch_size) array for output
        c = zeros(typeof(yhat[1][1]), out_seq_len, size(yhat, 2)) |> gpu

        for i in 1:out_seq_len
            # predict next
            yhat2 = modelPredictor(_yhat)
            c[i, :] = vec(modelDNN(yhat2))
            _yhat = yhat2
        end

        c
    end

    state = (modelCNN, modelTrainer, modelPredictor, modelDNN)

    # define model by combining multiple models
    # x : batched input for CNN ((length(features), sample_size, 1, batch_size))
    function model(x)
        # do CNN to extract local features
        ŷ_CNN = modelCNN(x)

        # 4D -> Array of 2Ds (hidCNN, batch)
        # [(hidCNN, batch), ...]
        x_RNN = unpack_seq(ŷ_CNN) |> gpu

        # RNN
        # `modelGRU.` performs GRU for input sequence
        # `predict_GRU` performs prediction after input seqeunce
        y_RNN_end = modelTrainer.(x_RNN)

        # copy hidden state from Trainer
        modelPredictor.state = modelTrainer.state

        # y_RNN_end[end] is a last sequence,
        ŷ_RNN = predict_recur(y_RNN_end[end], output_size)

        # Put into DNN and get result
        ŷ = modelDNN.(ŷ_RNN)

        Flux.reset!(state)

        ŷ_RNN
    end

    #loss(x, y) = Flux.mse(model(x), y) + sum(LinearAlgebra.norm, Flux.params(model))
    loss(x, y) = Flux.mse(model(x), y)
    # TODO : How to pass feature size
    accuracy(data) = RMSE(data, model, μσ)
    opt = Flux.ADAM()

    model, state, loss, accuracy, opt
end
