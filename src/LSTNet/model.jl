 """
    train(df, ycol, norm_prefix, norm_features,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        statvals, filename, test_dates)

"""
function train_LSTNet(train_wd::Array{DataFrame, 1}, valid_wd::Array{DataFrame, 1}, test_wd::Array{DataFrame, 1}, 
    ycol::Symbol, norm_prefix::String, features::Array{Symbol},
    train_size::Integer, valid_size::Integer, test_size::Integer,
    sample_size::Integer, batch_size::Integer, kernel_length::Integer,
    output_size::Integer, epoch_size::Integer, _eltype::DataType,
    statvals::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where I <: Integer

    @info "LSTNet training starts.."

    norm_ycol = Symbol(norm_prefix, ycol)
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]

    # extract from ndsparse
    total_μ = statvals[String(ycol), "μ"].value
    total_σ = statvals[String(ycol), "σ"].value
    total_min = float(statvals[String(ycol), "minimum"].value)
    total_max = float(statvals[String(ycol), "maximum"].value)

    # compute mean and std by each train/valid/test set
    statval = ndsparse((
        dataset = ["total", "total", "total", "total"],
        type = ["μ", "σ", "maximum", "minimum"]),
        (value = [total_μ, total_σ, total_max, total_min],))

    # construct compile function symbol
    compile = eval(Symbol(:compile, "_", ycol, "_LSTNet"))
    model, state, loss, accuracy, opt = 
        compile((length(features), kernel_length), sample_size, output_size, statval)

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Construct Training Set batch..."
    p = Progress(div(length(train_wd), batch_size) + 1, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, _eltype))
        for dfs in Base.Iterators.partition(train_wd, batch_size)]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, _eltype))
        for dfs in Base.Iterators.partition(valid_wd, batch_size)]

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    # batch_size should be 1 except raining
    
    test_set = [(ProgressMeter.next!(p);
        make_pair_LSTNet(df, norm_ycol, norm_features,
            sample_size, kernel_length, output_size, _eltype))
        for df in test_wd]
    #=
    test_set = [(ProgressMeter.next!(p);
        make_batch_LSTNet(collect(dfs), norm_ycol, norm_features,
            sample_size, kernel_length, output_size, batch_size, _eltype))
        for dfs in Base.Iterators.partition(test_wd, batch_size)]
    =#

    # *_set : normalized
    df_evals = train_LSTNet!(state, model, train_set, valid_set,
        loss, accuracy, opt, epoch_size, statval, norm_features, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    eval_metrics = [:RMSE, :MAE, :MSPE, :MAPE]

    # test set
    for metric in eval_metrics
        # pure test set is too slow on evaluation, 
        # batched test set is used only in evaluation
        _eval = evaluation2(test_set, model, statval, metric)
        @info " $(string(ycol)) $(string(metric)) for test   : ", _eval
    end

    # valid set
    for metric in eval_metrics
        _eval = evaluation2(valid_set, model, statval, metric)
        @info " $(string(ycol)) $(string(metric)) for valid  : ", _eval
    end

    # create directory per each time
    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath("/mnt/$(i_pad)/")
    end

    # batched set can't be used to create table (predict_model_* )
    rnn_table = predict_RNN_model_zscore(test_set, model, ycol, total_μ, total_σ, output_size, "/mnt/")
    dfs_out = export_CSV(DateTime.(test_dates, Local), rnn_table, ycol, output_size, "/mnt/", String(ycol))
    df_corr = compute_corr(rnn_table, output_size, "/mnt/", String(ycol))

    plot_DNN_scatter(rnn_table, ycol, output_size, "/mnt/", String(ycol))
    plot_DNN_histogram(rnn_table, ycol, output_size, "/mnt/", String(ycol))

    plot_datefmt = @dateformat_str "yyyymmddHH"

    plot_DNN_lineplot(DateTime.(test_dates, Local), rnn_table, ycol, output_size, "/mnt/", String(ycol))
    plot_corr(df_corr, output_size, "/mnt/", String(ycol))

    # 3 months plot
    # TODO : how to generalize date range? how to split based on test_dates?
    # 1/4 : because train size is 3 days, result should be start from 1/4
    # 12/29 : same reason 1/4, but this results ends with 12/31 00:00 ~ 12/31 23:00
    
    push!(eval_metrics, :learn_rate)
    plot_evaluation(df_evals, ycol, eval_metrics, "/mnt/")

    model, statval
end

function train_LSTNet!(state, model, train_set, valid_set, loss, accuracy, opt,
    epoch_size::Integer, statval::AbstractNDSparse, norm_features::Array{Symbol},
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
            evaluations2(valid_set, model, statval,
            [:RMSE, :MAE, :MSPE, :MAPE])
        push!(df_eval, [epoch_idx opt.eta _acc rmse mae mspe mape])

        # Calculate accuracy:
        _acc = cpu.(accuracy(valid_set))
        _loss = cpu.(loss(train_set[1][1], train_set[1][2], train_set[1][3]))
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
            weightsCNN = Flux.params(cpu_modelCNN)
            cpu_encoder = state[2] |> cpu
            weightsEncoder = Flux.params(cpu_encoder)
            cpu_decoder = state[3] |> cpu
            weightsDecoder = Flux.params(cpu_decoder)
            cpu_modelDNN = state[4] |> cpu
            weightsDNN = Flux.params(cpu_modelDNN)
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            μ, σ = statval["total", "μ"].value, statval["total", "σ"].value
            total_max, total_min =
                float(statval["total", "maximum"].value), float(statval["total", "minimum"].value)
            # BSON can't save weights now (2019/12), disable temporarily
            BSON.@save filepath cpu_modelCNN cpu_encoder cpu_decoder cpu_modelDNN epoch_idx _acc μ σ total_max total_min

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
    stackEncoded(_x, _size)

Convert CNH array to RNN style batch sequence
Use Zygote.Buffer to create adjoint for mutating array
"""
function stackEncoded(_x::AbstractArray{R, 3}, _size::Integer) where R <: Real
    # hidCNN x batch_size x sample_size
    # => [hidCNN x batch_size] x sample_size
    buf = Zygote.Buffer([_x[:, :, 1]], _size)
    for i = 1:_size
        buf[i] = _x[:, :, i]
    end
    copy(buf)
end

"""
    stackEncoded(_x, _size)

Extrapolate 2D array to RNN style batch sequence
Use Zygote.Buffer to create adjoint for mutating array

This might not needed because xd has been precomputed as batch sequences
"""
function stackDecoded(_x::AbstractArray{R, 2}, _size::Integer) where R <: Real
    # decoded tokens are replace by zeros of num_output x batch_size
    # num_output x batch_size
    # => [[num_output x batch_size]...] length is _size
    _x_3d = zeros(eltype(_x), size(_x, 1), size(_x)...)
    buf = Zygote.Buffer([_x_3d[:, 1, :]], _size)
    for i = 1:_size
        buf[i] = _x_3d[:, i, :]
    end
    copy(buf)
end

"""
    compile_PM10_LSTNetRecurSkip(kernel_size, sample_size, output_size)

kernel_size is 2D tuple for CNN kernel
sample_size and output_size is Integer

# Arguments
* x : dim_input x window_size x 1 x batch_size

# Returns
* num_output x batch_size
"""
function compile_PM10_LSTNet(
    kernel_size::Tuple{I, I},
    sample_size::I, output_size::I,
    statval::AbstractNDSparse) where I<:Integer

    @info("    Compiling PM10 model...")
    # hidCNN : length of CNN latent dim, CNN channel
    # hidRNN : length of RNN latent dim
    hidCNN = 8
    hidRNN = 16

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
    #   * Kernel Size : (kernel_length, length(features))
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidRNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidCNN, leakyrelu)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidRNN, batch_size),...]
    modelGRU1 = GRU(hidCNN, hidRNN) |> gpu
    modelGRU2 = GRU(output_size, hidRNN) |> gpu

    modelLSTM1 = LSTM(hidCNN, hidRNN) |> gpu
    modelLSTM2 = LSTM(output_size, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Dense(hidRNN, output_size) |> gpu

    state = (
        CNNmodel = modelCNN,
        encoder = modelLSTM1,
        decoder = modelLSTM2,
        DNNmodel = modelDNN)

    # Define model by combining multiple models
    # Good Tutorial : https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction
    # Arguments
    #   xe : batched input for model (sample_size, (length(features), 1, batch_size))
    #   xd : batched sequence for decoder [(output_size, batch_size)...] (output_size,)
    # Returns
    #   y_hat : prediction result (output_size, batch_size)
    function model(xe, xd)
        # CNN to extract local features
        # size(x_cnn) == (1, sample_size, hidCNN, batch_size)
        x_whcn = state.CNNmodel(xe)

        # 4D (WHCN) (1, sample_size, hidCNN, batch_size) -> 
        # 3D Array (hidCNN, sample_size, batch_size)
        x_cnh = whcn2cnh(x_whcn)

        # RNN
        Flux.reset!(state.encoder)
		Flux.reset!(state.decoder)

        # CNH array to batch sequences
        # 3D Array (hidCNN, sample_size, batch_size) ->
        # Array of Array [(hidCNN, batch_size)...]:  (sample_size,)
        x_encoded = stackEncoded(x_cnh, sample_size)

        # 2D Array to batch sequences (all zeros)
        # 2D Array (output_size, batch_size) ->
        # Array of Array [(output_size, batch_size)...]:  (output_size,)
		# x_decoded = stackDecoded(xd, output_size)

        # broadcast : train columns. Each column (single batch) have been trained independently
        # drop _encoded, only need state from encoder
        _encoded = state.encoder.(x_encoded)

        # copy state from encoder to decoder
        state.decoder.state = state.encoder.state

        #_decoded = state.decoder.(x_decoded)
        # precomputed decoded input
        _decoded = state.decoder.(xd)
		y_decoded = state.decoder.state[2]

        # each seq element represzents hidden state at each time stamp
		# what I want
		# In: hidCNN x window_size x batch_size
		# 1: [hidCNN x batch_size_size] x window_size (stackEncoded & encoder)
		# 2: [hidRNN x batch_size_size] x window_size (stackDecoded & decoder)
		# 3: num_output x batch_size (Dense)
		# Out: num_output x batch_size
        state.DNNmodel(y_decoded)
    end

    loss(xe, xd, y) = Flux.mse(model(xe, xd), y)
    accuracy(data) = evaluation2(data, model, statval, :RMSE)

    opt = Flux.ADAM()

    @info "hidCNN       in PM10: ", hidCNN
    @info "hidRNN       in PM10: ", hidRNN
    @info "ModelCNN     in PM10: ", state.CNNmodel
    @info "Encoder(RNN) in PM10: ", state.encoder
    @info "Decoder(RNN) in PM10: ", state.decoder
    @info "ModelDNN     in PM10: ", state.DNNmodel
    model, state, loss, accuracy, opt
end

function compile_PM25_LSTNet(
    kernel_size::Tuple{I, I},
    sample_size::I, output_size::I,
    statval::AbstractNDSparse) where I<:Integer

    @info("    Compiling PM25 model...")
    # hidCNN : length of CNN latent dim, CNN channel
    # hidRNN : length of RNN latent dim
    hidCNN = 8
    hidRNN = 16

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
    #   * Kernel Size : (kernel_length, length(features))
    #   * Input Size : (sample_size + pad_sample_size, length(features), 1, batch_size)
    #   * Output Size : (sample_size, 1, hidRNN, batch_size)
    modelCNN = Chain(
        Conv(kernel_size, 1 => hidCNN, leakyrelu)) |> gpu
    
    # 2. GRU
    #   * Input shape : (sample_size, 1, hidCNN, batch_size) ->
    #                   [(hidRNN, batch_size),...]
    modelGRU1 = GRU(hidCNN, hidRNN) |> gpu
    modelGRU2 = GRU(output_size, hidRNN) |> gpu

    modelLSTM1 = LSTM(hidCNN, hidRNN) |> gpu
    modelLSTM2 = LSTM(output_size, hidRNN) |> gpu

    # DNN converts GRU output to single real output value
    modelDNN = Dense(hidRNN, output_size) |> gpu

    state = (
        CNNmodel = modelCNN,
        encoder = modelLSTM1,
        decoder = modelLSTM2,
        DNNmodel = modelDNN)

    # Define model by combining multiple models
    # Good Tutorial : https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction
    # Arguments
    #   xe : batched input for model (sample_size, (length(features), 1, batch_size))
    #   xd : batched sequence for decoder [(output_size, batch_size)...] (output_size,)
    # Returns
    #   y_hat : prediction result (output_size, batch_size)
    function model(xe, xd)
        # CNN to extract local features
        # size(x_cnn) == (1, sample_size, hidCNN, batch_size)
        x_CNN = state.CNNmodel(xe)

        # 4D (WHCN) (1, sample_size, hidCNN, batch_size) -> 
        # 3D Array (hidCNN, sample_size, batch_size)
        x_cnh = whcn2cnh(x_CNN)

        # RNN
        Flux.reset!(state.encoder)
		Flux.reset!(state.decoder)

        # CNH array to batch sequences
        # 3D Array (hidCNN, sample_size, batch_size) ->
        # Array of Array [(hidCNN, batch_size)...]:  (sample_size,)
        x_encoded = stackEncoded(x_cnh, sample_size)

        # 2D Array to batch sequences (all zeros)
        # 2D Array (output_size, batch_size) ->
        # Array of Array [(output_size, batch_size)...]:  (output_size,)
		# x_decoded = stackDecoded(xd, output_size)

        # broadcast : train columns. Each column (single batch) have been trained independently
        # drop _encoded, only need state from encoder
        _encoded = state.encoder.(x_encoded)

        # copy state from encoder to decoder
        state.decoder.state = state.encoder.state

        #_decoded = state.decoder.(x_decoded)
        # precomputed decoded input
        _decoded = state.decoder.(xd)
		y_decoded = state.decoder.state[2]

        # each seq element represzents hidden state at each time stamp
		# what I want
		# In: hidCNN x window_size x batch_size
		# 1: [hidCNN x batch_size_size] x window_size (stackEncoded & encoder)
		# 2: [hidRNN x batch_size_size] x window_size (stackDecoded & decoder)
		# 3: num_output x batch_size (Dense)
		# Out: num_output x batch_size
        state.DNNmodel(y_decoded)
    end

    loss(xe, xd, y) = Flux.mse(model(xe, xd), y)
    accuracy(data) = evaluation2(data, model, statval, :RMSE)

    opt = Flux.ADAM()
    
    @info "hidCNN       in PM25: ", hidCNN
    @info "hidRNN       in PM25: ", hidRNN
    @info "ModelCNN     in PM25: ", state.CNNmodel
    @info "Encoder(RNN) in PM25: ", state.encoder
    @info "Decoder(RNN) in PM25: ", state.decoder
    @info "ModelDNN     in PM25: ", state.DNNmodel
    model, state, loss, accuracy, opt
end
