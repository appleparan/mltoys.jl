data_(x) = vcat(x...)
tdata_(x) = Tracker.data(vcat(x...))

"""
    train(df, ycol, norm_prefix, norm_features,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        μσs, filename, test_dates)

"""
function train_LSTNet(train_wd::Array{DataFrame, 1}, valid_wd::Array{DataFrame, 1}, test_wd::Array{DataFrame, 1}, 
    ycol::Symbol, norm_prefix::String, features::Array{Symbol},
    train_size::Integer, valid_size::Integer, test_size::Integer,
    sample_size::Integer, batch_size::Integer, kernel_length::Integer,
    output_size::Integer, epoch_size::Integer, eltype::DataType,
    μσs::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where I <: Integer

    @info "LSTNet training starts.."

    norm_ycol = Symbol(norm_prefix, ycol)
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]

    # extract from ndsparse
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value
    total_min = float(μσs[String(ycol), "minimum"].value)
    total_max = float(μσs[String(ycol), "maximum"].value)

    # compute mean and std by each train/valid/test set
    μσ = ndsparse((
        dataset = ["total", "total", "total", "total"],
        type = ["μ", "σ", "maximum", "minimum"]),
        (value = [total_μ, total_σ, total_max, total_min],))

    # construct compile function symbol
    compile = eval(Symbol(:compile, "_", ycol, "_LSTNet"))
    model, state, loss, accuracy, opt = 
        compile((kernel_length, length(features)), batch_size, output_size, μσ)

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Construct Training Set batch..."
    p = Progress(div(length(train_wd), batch_size) + 1, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, eltype))
        for dfs in Base.Iterators.partition(train_wd, batch_size)]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, eltype))
        for dfs in Base.Iterators.partition(valid_wd, batch_size)]

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    # batch_size should be 1 except raining
    test_set = [(ProgressMeter.next!(p);
        make_pair_LSTNet(df, norm_ycol, norm_features,
            sample_size, kernel_length, output_size, eltype))
        for df in test_wd]
    test_batch_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, eltype))
        for dfs in Base.Iterators.partition(test_wd, batch_size)]

    # *_set : normalized
    df_evals = train_LSTNet!(state, model, train_set, valid_set, loss, accuracy, opt, epoch_size, μσ, norm_features, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    eval_metrics = [:RMSE, :MAE, :MSPE, :MAPE]
    
    # test set evaluation
    for metric in eval_metrics
        # pure test set is too slow on evaluation, 
        # batched test set is used only in evaluation
        metric_func = :($(metric)($(test_batch_set), $(model), $(μσ), $(tdata_)))
        @info " $(string(ycol)) $(string(metric)) for test   : ", eval(metric_func)
    end

    # valid set evaluation
    for metric in eval_metrics
        metric_func = :($(metric)($(valid_set), $(model), $(μσ), $(tdata_)))
        @info " $(string(ycol)) $(string(metric)) for valid  : ", eval(metric_func)
    end

    # create directory per each time
    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath("/mnt/$(i_pad)/")
    end

    # batched set can't be used to create table (predict_model_* )
    dnn_table = predict_RNNmodel_zscore(test_set, model, ycol, total_μ, total_σ, output_size, "/mnt/", tdata_)
    #dnn_table = predict_RNNmodel_minmax(test_set, model, ycol, total_min, total_max, 0.0, 10.0, output_size, "/mnt/", tdata_)
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
    epoch_size::Integer, μσ::AbstractNDSparse, norm_features::Array{Symbol},
    filename::String) where {F <: AbstractFloat, T2 <: Tuple{AbstractArray{F, 2}, AbstractArray{F, 2}},
    T1 <: Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}}}

    @info("    Beginning training loop...")
    flush(stdout); flush(stderr)

    _acc = Inf
    # for adjR2 -∞ is the worst
    best_acc = Inf
    last_improvement = 0

    df_eval = DataFrame(epoch = Int64[], learn_rate = Float64[], ACC = Float64[],
        RMSE = Float64[], MAE = Float64[], MSPE = Float64[], MAPE = Float64[])

    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        # train model with normalized data set
        Flux.train!(loss, Flux.params(state), train_set, opt)

        # record evaluation
        rmse, mae, mspe, mape =
            evaluations(valid_set, model, μσ, norm_features,
            [:RMSE, :MAE, :MSPE, :MAPE], tdata_)
        push!(df_eval, [epoch_idx opt.eta _acc rmse mae mspe mape])

        # Calculate accuracy:
        _acc = accuracy(valid_set)
        _loss = Tracker.data(loss(train_set[1][1], train_set[1][2]))
        @info(@sprintf("epoch [%d]: loss[1]: %.8E Valid accuracy: %.8f Time: %s", epoch_idx, _loss, _acc, now()))
        flush(stdout); flush(stderr)

        if _acc < 0.001
            @info("    -> Early-exiting: We reached our target accuracy")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if _acc < best_acc
            @info "    -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_modelCNN = state[1] |> cpu
            weightsCNN = Tracker.data.(Flux.params(cpu_modelCNN))
            cpu_modelGRU = state[2] |> cpu
            weightsGRU = Tracker.data.(Flux.params(cpu_modelGRU))
            cpu_modelDNN = state[3] |> cpu
            weightsDNN = Tracker.data.(Flux.params(cpu_modelDNN))
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            μ, σ = μσ["total", "μ"].value, μσ["total", "σ"].value
            total_max, total_min =
                float(μσ["total", "maximum"].value), float(μσ["total", "minimum"].value)
            BSON.@save filepath cpu_modelCNN weightsCNN cpu_modelGRU weightsGRU cpu_modelDNN weightsDNN epoch_idx _acc μ σ total_max total_min
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
    sample_size::I, output_size::I,
    μσ::AbstractNDSparse) where I<:Integer

    @info("    Compiling model...")
    # hidRNN : length of context vector
    hidRNN = 32

    kernel_length = kernel_size[1]
    # WHCN order : reverse to python framework (row-major && column-major)
    # W : Width = # of variables (= # of features)
    # H : Height = # of timesteps
    # C : Channel = 1
    # N : batches

    # 1. CNN
    #   extract short-term patterns in the time dimension 
    #   as well as local dependencies between variables
    #   * Channel : 1 => hidRNN 
    #   * Kernel Size : (kernel_length, length(features)
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidRNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidRNN, leakyrelu),
        Dropout(0.2)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidRNN, batch_size),...]
    # modelGRU is a Single GRU Cell
    # hidCNN is a embedding size in NLP
    # sample_size is a sequnce length in NLP
    # Output has same dimension as input
    #
    # Check figure in 'Build the Model' Section in 
    # https://www.tensorflow.org/tutorials/text/text_generation
    # SEQ_LENGTH = sample_size
    # Char Embdding = hidRNN
    # logits = output value itself
    modelGRU = GRU(hidRNN, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Chain(
        Dense(hidRNN, 1)) |> gpu

    function predict_recur(_yhat, seq_len::Integer)
        # _yhat : current output of RNN (length : hidRNN)
        # size(_yhat) = (hidRNN x batch_size)
        yhat = _yhat
        # (seq_len x batch_size) array for output
        r = []

        for _ in 1:seq_len
            # predict next output by giving input as one by one
            yhat2 = modelGRU(yhat)
            push!(r, modelDNN(yhat2))
            yhat = yhat2
        end

        r
    end

    state = (modelCNN, modelGRU, modelDNN)

    # define model by combining multiple models
    # x : batched input for CNN (sample_size, (length(features), 1, batch_size))
    function model(x)
        # to avoid segfault due to GC
        GC.@preserve modelCNN modelGRU modelDNN

        # do CNN to extract local features
        ŷ_CNN = modelCNN(x)

        # 4D -> Array of 2Ds (hidCNN, batch)
        x_RNN = unpack_seq(ŷ_CNN)

        # RNN
        # `modelGRU.` performs GRU for input sequence
        # `predict_GRU` performs prediction after input seqeunce
        Flux.reset!(modelGRU)
        # broadcast : train columns. Each column (single batch) have been trained
        # @test trained[end] == modelGRU.state # true
        trained = modelGRU.(x_RNN)

        # predict
        ŷ_RNN = predict_recur(modelGRU.state, output_size)

        ŷ_RNN
    end

    loss(x, y) = Flux.mse(data_(model(x)), y)

    accuracy(data) = RMSE(data, model, μσ, tdata_)
    opt = Flux.ADAM()

    @info "hidRNN       in PM10: ", hidRNN
    @info "ModelCNN     in PM10: ", modelCNN
    @info "ModelGRU     in PM10: ", modelGRU
    @info "ModelDNN     in PM10: ", modelDNN
    model, state, loss, accuracy, opt
end

function compile_PM25_LSTNet(
    kernel_size::Tuple{I, I},
    sample_size::I, output_size::I,
    μσ::AbstractNDSparse) where I<:Integer

    @info("    Compiling model...")
    # hidRNn : length of context vector
    hidRNN = 32

    kernel_length = kernel_size[1]
    # WHCN order : reverse to python framework (row-major && column-major)
    # W : Width = # of variables (= # of features)
    # H : Height = # of timesteps
    # C : Channel = 1
    # N : batches

    # 1. CNN
    #   extract short-term patterns in the time dimension 
    #   as well as local dependencies between variables
    #   * Channel : 1 => hidRNN 
    #   * Kernel Size : (kernel_length, length(features)
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidRNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidRNN, leakyrelu),
        Dropout(0.2)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidRNN, batch_size),...]
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
    modelGRU = GRU(hidRNN, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Chain(
        Dense(hidRNN, 1)) |> gpu

    function predict_recur(_yhat, seq_len::Integer)
        # _yhat : current output of RNN (length : hidRNN)
        # size(_yhat) = (hidRNN x batch_size)
        yhat = _yhat
        # (seq_len x batch_size) array for output
        r = []

        for _ in 1:seq_len
            # predict next output by giving input as one by one
            yhat2 = modelGRU(yhat)
            push!(r, modelDNN(yhat2))
            yhat = yhat2
        end

        r
    end

    state = (modelCNN, modelGRU, modelDNN)

    # define model by combining multiple models
    # x : batched input for CNN (sample_size, (length(features), 1, batch_size))
    function model(x)
        # to avoid segfault due to GC
        GC.@preserve modelCNN modelGRU modelDNN

        # do CNN to extract local features
        ŷ_CNN = modelCNN(x)

        # 4D -> Array of 2Ds (hidCNN, batch)
        x_RNN = unpack_seq(ŷ_CNN)

        # RNN
        # `modelGRU.` performs GRU for input sequence
        # `predict_GRU` performs prediction after input seqeunce
        Flux.reset!(modelGRU)
        # broadcast : train columns. Each column (single batch) have been trained
        # @test trained[end] == modelGRU.state # true
        trained = modelGRU.(x_RNN)

        # predict
        ŷ_RNN = predict_recur(modelGRU.state, output_size)

        ŷ_RNN
    end

    loss(x, y) = Flux.mse(data_(model(x)), y)

    accuracy(data) = RMSE(data, model, μσ, tdata_)
    opt = Flux.ADAM()
    
    @info "hidRNN       in PM25: ", hidRNN
    @info "ModelCNN     in PM25: ", modelCNN
    @info "ModelGRU     in PM25: ", modelGRU
    @info "ModelDNN     in PM25: ", modelDNN
    model, state, loss, accuracy, opt
end
