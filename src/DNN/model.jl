"""
    train(df, ycol, scaled_prefix, scaled_features,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        statvals, filename, test_dates)

"""
function train_DNN(train_wd::Array{DataFrame, 1}, valid_wd::Array{DataFrame, 1}, test_wd::Array{DataFrame, 1}, 
    ycol::Symbol, scaled_ycol::Symbol, scaled_features::Array{Symbol}, scaling_method::Symbol,
    train_size::Integer, valid_size::Integer, test_size::Integer,
    sample_size::Integer, input_size::Integer, batch_size::Integer, output_size::Integer,
    epoch_size::Integer, eltype::DataType,
    test_dates::Array{ZonedDateTime,1},
    statvals::AbstractNDSparse, season_table::AbstractNDSparse,
    output_prefix::String, filename::String) where I <: Integer

    @info "DNN training starts.."

    # extract from ndsparse
    total_μ = statvals[String(ycol), "μ"].value
    total_σ = statvals[String(ycol), "σ"].value
    total_max = float(statvals[String(ycol), "maximum"].value)
    total_min = float(statvals[String(ycol), "minimum"].value)

    # compute mean and std by each train/valid/test set
    statval = ndsparse((
        dataset = ["total", "total", "total", "total"],
        type = ["μ", "σ", "maximum", "minimum"]),
        (value = [total_μ, total_σ, total_max, total_min],))

    # construct compile function symbol
    compile = eval(Symbol(:compile, "_", ycol, "_DNN"))
    model, loss, accuracy, opt = compile(input_size, batch_size, output_size, statval)

    # modify scaled_features to train residuals

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Construct Training Set batch..."
    p = Progress(div(length(train_wd), batch_size) + 1, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_DNN(collect(dfs), scaled_ycol, scaled_features,
            sample_size, output_size, batch_size, 0.5, eltype))
        for dfs in Base.Iterators.partition(train_wd, batch_size)]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, scaled_ycol, scaled_features, sample_size, output_size, eltype)) for df in valid_wd]

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, scaled_ycol, scaled_features, sample_size, output_size, eltype)) for df in test_wd]

    train_set = gpu.(train_set)
    valid_set = gpu.(valid_set)
    test_set = gpu.(test_set)

    # *_set : normalized values
    df_evals = train_DNN!(model, train_set, valid_set,
        loss, accuracy, opt, epoch_size, statval, scaled_features, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    eval_metrics = [:RMSE, :MAE, :MSPE, :MAPE]
    # test set
    for metric in eval_metrics
        _eval = evaluation(test_set, model, statval, metric)
        @info " $(string(ycol)) $(string(metric)) for test   : ", _eval
    end

    # valid set
    for metric in eval_metrics
        _eval = evaluation(valid_set, model, statval, metric)
        @info " $(string(ycol)) $(string(metric)) for valid  : ", _eval
    end

    # create directory per each time
    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath("/$(output_prefix)/$(i_pad)/")
    end

    # back to unnormalized for comparison
    if scaling_method == :zscore
        dnn_table = predict_DNN_model_zscore(test_set, model, ycol,
            total_μ, total_σ, output_size, "/$(output_prefix)/")
    elseif scaling_method == :minmax
        dnn_table = predict_DNN_model_minmax(test_set, model, ycol,
            total_min, total_max, 0.0, 10.0, output_size, "/$(output_prefix)/")
    elseif scaling_method == :logzscore
        log_μ = statvals[string(:log_, ycol), "μ"].value
        log_σ = statvals[string(:log_, ycol), "σ"].value
        dnn_table = predict_DNN_model_logzscore(test_set, model, ycol,
            log_μ, log_σ, output_size, "/$(output_prefix)/")
    elseif scaling_method == :invzscore
        inv_μ = statvals[string(:inv_, ycol), "μ"].value
        inv_σ = statvals[string(:inv_, ycol), "σ"].value
        dnn_table = predict_DNN_model_invzscore(test_set, model, ycol,
            inv_μ, inv_σ, output_size, "/$(output_prefix)/")
    elseif scaling_method == :logminmax
        log_max = statvals[string(:log_, ycol), "maximum"].value
        log_min = statvals[string(:log_, ycol), "minimum"].value
        dnn_table = predict_DNN_model_logminmax(test_set, model, ycol,
            log_min, log_max, 0.0, 10.0, output_size, "/$(output_prefix)/")
    end

    # 3 months plot
    # TODO : how to generalize date range? how to split based on test_dates?
    # 1/4 : because train size is 3 days, result should be start from 1/4
    # 12/29 : same reason 1/4, but this results ends with 12/31 00:00 ~ 12/31 23:00
    push!(eval_metrics, :learn_rate)
    plot_evaluation(df_evals, ycol, eval_metrics, "/$(output_prefix)/")

    model, dnn_table, statval
end

function train_DNN!(model::C,
    train_set::Array{T2, 1}, valid_set::Array{T1, 1},
    loss, accuracy, opt,
    epoch_size::Integer, statval::AbstractNDSparse, scaled_features::Array{Symbol},
    filename::String) where {C <: Flux.Chain, F <: AbstractFloat, T2 <: Tuple{AbstractArray{F, 2}, AbstractArray{F, 2}},
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
        Flux.train!(loss, Flux.params(model), train_set, opt)

        # record evaluation
        rmse, mae, mspe, mape =
            evaluations(valid_set, model, statval,
            [:RMSE, :MAE, :MSPE, :MAPE])
        push!(df_eval, [epoch_idx opt.eta _acc rmse mae mspe mape])

        # Calculate accuracy:
        _acc = accuracy(valid_set) |> cpu
        _loss = loss(train_set[1][1], train_set[1][2]) |> cpu
        @info(@sprintf("epoch [%d]: loss[1]: %.8E Valid accuracy: %.8f Time: %s", epoch_idx, _loss, _acc, now()))
        flush(stdout); flush(stderr)

        # If our accuracy is good enough, quit out.
        if _acc < 0.001
            @info("    -> Early-exiting: We reached our target accuracy")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if _acc < best_acc
            @info "    -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_model = model |> cpu
            weights = Flux.params(cpu_model)
            filepath = "/mnt/" * filename * ".bson"
            μ, σ = statval["total", "μ"].value, statval["total", "σ"].value
            total_max, total_min =
                float(statval["total", "maximum"].value), float(statval["total", "minimum"].value)
            # BSON can't save weights now (2019/12), disable temporarily
            BSON.@save filepath cpu_model epoch_idx _acc μ σ total_max total_min

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

function compile_PM10_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, statval::AbstractNDSparse)
    @info("    Compiling model...")

    unit_size = 16

    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    # not sigmoid, elu used to reduce vanishing gradient problem
    # predict low concentration is not important than high concentration
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    GC.@preserve model

    @info "Unit size in PM10: ", unit_size
    @info "Model     in PM10: ", model

    # no regularization
    loss(x, y) = Flux.mse(model(x), y)

    # L1
    #loss(x, y) = Flux.mse(model(x), y) +
    #    lambda(x, y) *
    #    sum(x1 -> LinearAlgebra.norm(x1, 1), Flux.params(model))

    # L2 
    # disable due to Flux#930 issue
    #lambda(x, y) = 10^(log10(Flux.mse(model(x), y)) - 1)
    #loss(x, y) = Flux.mse(model(x), y) + lambda(x, y) * sum(LinearAlgebra.norm, Flux.params(model))

    # Changhoon lee's method to capture outliers
    #loss(x, y) = sum(exp.(y .* 0.000001)) * Flux.mse(model(x), y)
    #loss(x, y) = maximum(10.0.^(y .* 0.0001)) * Flux.mse(model(x), y)

    accuracy(data) = evaluation(data, model, statval, :RMSE)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_PM25_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, statval::AbstractNDSparse)
    @info("    Compiling model...")

    unit_size = 32

    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    # not sigmoid, elu used to reduce vanishing gradient problem
    # predict low concentration is not important than high concentration
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    GC.@preserve model

    @info "Unit size in PM25: ", unit_size
    @info "Model     in PM25: ", model

    # no regularization
    loss(x, y) = Flux.mse(model(x), y)

    # L1 regularization
    #loss(x, y) = Flux.mse(model(x), y) +
    #    lambda(x, y) *
    #    sum(x1 -> LinearAlgebra.norm(x1, 1), Flux.params(model))

    # L2 regularization
    # disable regularization due to Flux#930 issue
    #lambda(x, y) = 10^(log10(Flux.mse(model(x), y)) - 1)
    #loss(x, y) = Flux.mse(model(x), y) + lambda(x, y) * sum(LinearAlgebra.norm, Flux.params(model))

    # Changhoon lee's method to capture outliers
    #loss(x, y) = sum(exp.(y .* 0.000001)) * Flux.mse(model(x), y)
    #loss(x, y) = maximum(10.0.^(y .* 0.0001)) * Flux.mse(model(x), y)

    accuracy(data) = evaluation(data, model, statval, :RMSE)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end
