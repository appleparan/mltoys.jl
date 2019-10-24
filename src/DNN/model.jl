"""
    train(df, ycol, norm_prefix, norm_feas,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        μσs, filename, test_dates)

"""
function train_DNN(train_wd::Array{DataFrame, 1}, valid_wd::Array{DataFrame, 1}, test_wd::Array{DataFrame, 1}, 
    ycol::Symbol, norm_prefix::String, feas::Array{Symbol},
    train_size::Integer, valid_size::Integer, test_size::Integer,
    sample_size::Integer, input_size::Integer, batch_size::Integer, output_size::Integer,
    epoch_size::Integer, eltype::DataType,
    μσs::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where I <: Integer

    @info "DNN training starts.."

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
    compile = eval(Symbol(:compile, "_", ycol, "_DNN"))
    model, loss, accuracy, opt = compile(input_size, batch_size, output_size, μσ)

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704
    @info "    Construct Training Set batch..."
    p = Progress(div(length(train_wd), batch_size) + 1, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_DNN(collect(dfs), norm_ycol, norm_feas,
            sample_size, output_size, batch_size, 0.5, eltype))
        for dfs in Base.Iterators.partition(train_wd, batch_size)]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, norm_ycol, norm_feas, sample_size, output_size, eltype)) for df in valid_wd]

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, norm_ycol, norm_feas, sample_size, output_size, eltype)) for df in test_wd]

    # *_set : normalized
    df_evals = train_DNN!(model, train_set, valid_set, loss, accuracy, opt, epoch_size, μσ, norm_feas, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    eval_metrics = [:RMSE, :MAE, :MSPE, :MAPE]
    # test set
    for metric in eval_metrics
        metric_func = :($(metric)($(test_set), $(model), $(μσ)))
        @info " $(string(ycol)) $(string(metric)) for test   : ", eval(metric_func)
    end

    # valid set
    for metric in eval_metrics
        metric_func = :($(metric)($(valid_set), $(model), $(μσ)))
        @info " $(string(ycol)) $(string(metric)) for valid  : ", eval(metric_func)
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

function train_DNN!(model::C,
    train_set::Array{T2, 1}, valid_set::Array{T1, 1},
    loss, accuracy, opt,
    epoch_size::Integer, μσ::AbstractNDSparse, norm_feas::Array{Symbol},
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
            evaluations(valid_set, model, μσ, norm_feas,
            [:RMSE, :MAE, :MSPE, :MAPE])
        push!(df_eval, [epoch_idx opt.eta _acc rmse mae mspe mape])

        # Calculate accuracy:
        _acc = Tracker.data(accuracy(valid_set))
        _loss = Tracker.data(loss(train_set[1][1], train_set[1][2]))
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
            weights = Tracker.data.(Flux.params(cpu_model))
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            μ, σ = μσ["total", "μ"].value, μσ["total", "σ"].value
            BSON.@save filepath cpu_model weights epoch_idx _acc μ σ
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

function compile_PM10_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ::AbstractNDSparse)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 3/3)), 768)
    unit_size = 16
    #unit_size = Int(round(input_size * 0.33))
    @show "Unit size in PM10: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

    # not sigmoid, elu used to reduce vanishing gradient problem
    # predict low concentration is not important than high concentration
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    #loss(x, y) = Flux.mse(model(x), y) + sum(LinearAlgebra.norm, Flux.params(model))
    loss(x, y) = Flux.mse(model(x), y)
    # TODO : How to pass feature size
    accuracy(data) = RMSE(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_PM25_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ::AbstractNDSparse)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 2/3)), 512)
    unit_size = 16
    #unit_size = Int(round(input_size * 0.33))
    @show "Unit size in PM25: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu),

        Dense(unit_size, unit_size, leakyrelu),

        Dense(unit_size, output_size)
    ) |> gpu

    #loss(x, y) = Flux.mse(model(x), y) + sum(LinearAlgebra.norm, Flux.params(model))
    loss(x, y) = Flux.mse(model(x), y)
    # TODO : How to pass feature size
    accuracy(data) = RMSE(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end
