"""
    train(df, ycol, norm_prefix, norm_feas,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        μσs, filename, test_dates)

"""
function train_DNN(df::DataFrame, ycol::Symbol, norm_prefix::String, _norm_feas::Array{Symbol},
    sample_size::Integer, input_size::Integer, batch_size::Integer, output_size::Integer, epoch_size::Integer,
    default_FloatType::DataType,
    train_valid_wd_idxs::Array{<:UnitRange{I}, 1}, test_wd_idxs::Array{<:UnitRange{I}, 1},
    train_chnk::Array{<:Array{<:UnitRange{I}, 1}, 1},
    train_idxs::Array{<:UnitRange{I}, 1}, valid_idxs::Array{<:UnitRange{I}, 1}, test_idxs::Array{<:UnitRange{I}, 1},
    μσs::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where I <: Integer

    @info "DNN training starts.."

    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)

    # extract from ndsparse
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value

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
    p = Progress(length(train_chnk), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    train_set = [(ProgressMeter.next!(p);
        make_batch_DNN(df, ycol, chnk, norm_feas,
        sample_size, output_size, batch_size, 0.5, default_FloatType)) for chnk in train_chnk]

    # don't construct minibatch for valid & test sets
    @info "    Construct Valid Set..."
    p = Progress(length(valid_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    valid_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, norm_ycol, idx, norm_feas, sample_size, output_size)) for idx in valid_idxs]

    @info "    Construct Test Set..."
    p = Progress(length(test_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, norm_ycol, idx, norm_feas, sample_size, output_size)) for idx in test_idxs]

    df_evals = train_DNN!(model, train_set, valid_set, loss, accuracy, opt, epoch_size, μσ, filename)
    
    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Test ACC  : ", accuracy(test_set)
    @info "    Valid ACC : ", accuracy(valid_set)
    flush(stdout); flush(stderr)

    @info " $(string(ycol)) RMSE for test   : ", RMSE(test_set, model, μσ)
    @info " $(string(ycol)) RSR for test    : ", RSR(test_set, model, μσ)
    @info " $(string(ycol)) PBIAS for test  : ", PBIAS(test_set, model, μσ)
    @info " $(string(ycol)) NSE for test    : ", NSE(test_set, model, μσ)
    @info " $(string(ycol)) IOA for test    : ", IOA(test_set, model, μσ)

    @info " $(string(ycol)) RMSE for valid  : ", RMSE(valid_set, model, μσ)
    @info " $(string(ycol)) RSR for valid   : ", RSR(valid_set, model, μσ)
    @info " $(string(ycol)) PBIAS for valid : ", PBIAS(valid_set, model, μσ)
    @info " $(string(ycol)) NSE for valid   : ", NSE(valid_set, model, μσ)
    @info " $(string(ycol)) IOA for valid   : ", IOA(valid_set, model, μσ)
    if ycol == :PM10 || ycol == :PM25
        forecast_all, forecast_high = classification(test_set, model, ycol)
        @info " $(string(ycol)) Forecasting accuracy (all) for test : ", forecast_all
        @info " $(string(ycol)) Forecasting accuracy (high) for test : ", forecast_high
    end

    table_01h, table_24h = compute_prediction(test_set, model, ycol, total_μ, total_σ, "/mnt/")
    y_01h_vals, ŷ_01h_vals, y_24h_vals, ŷ_24h_vals =
        export2CSV(DateTime.(test_dates), table_01h, table_24h, ycol, "/mnt/", string(ycol) * "_")
    plot_DNN_scatter(table_01h, table_24h, ycol, "/mnt/")
    plot_DNN_histogram(table_01h, table_24h, ycol, "/mnt/")

    plot_datefmt = @dateformat_str "yyyymmddHH"
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h, ycol, "/mnt/", String(ycol))

    _corr_01h = Statistics.cor(y_01h_vals, ŷ_01h_vals)
    _corr_24h = Statistics.cor(y_24h_vals, ŷ_24h_vals)
    @info " $(string(ycol)) Corr(01H)   : ", _corr_01h
    @info " $(string(ycol)) Corr(24H)   : ", _corr_24h

    # 3 months plot
    # TODO : how to generalize date range? how to split based on test_dates?
    # 1/4 : because train size is 3 days, result should be start from 1/4
    # 12/29 : same reason 1/4, but this results ends with 12/31 00:00 ~ 12/31 23:00

    plot_evaluation(df_evals, ycol, "/mnt/")

    model, μσ
end

function train_DNN!(model, train_set, valid_set, loss, accuracy, opt, epoch_size::Integer, μσ, filename::String)

    @info("    Beginning training loop...")
    flush(stdout); flush(stderr)

    best_acc = 100.0
    last_improvement = 0
    acc = 0.0

    df_eval = DataFrame(epoch = Int64[], learn_rate = Float64[], ACC = Float64[],
        RMSE = Float64[], RSR = Float64[], NSE = Float64[], PBIAS = Float64[], IOA = Float64[])

    for epoch_idx in 1:epoch_size
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Calculate accuracy:
        acc = accuracy("valid", valid_set)
        @info(@sprintf("epoch [%d]: Valid accuracy: %.6f Time: %s", epoch_idx, acc, now()))
        flush(stdout); flush(stderr)

        # record evaluation 
        #=
        rsr = RSR("valid", valid_set, model, μσ)
        nse = NSE("valid", valid_set, model, μσ)
        pbias = PBIAS("valid", valid_set, model, μσ)
        =#
        rmse, rsr, nse, pbias, ioa = evaluations(valid_set, model, μσ, [:RMSE, :RSR, :NSE, :PBIAS, :IOA])
        push!(df_eval, [epoch_idx opt.eta acc rmse rsr nse pbias ioa])

        # If our accuracy is good enough, quit out.
        if acc < 0.1
            @info("    -> Early-exiting: We reached our target accuracy (RSR) of 0.1")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc < best_acc
            @info "    -> New best accuracy! Saving model out to " * filename
            flush(stdout)

            cpu_model = model |> cpu
            weights = Tracker.data.(params(cpu_model))
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            BSON.@save filepath cpu_model weights epoch_idx acc
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

    df_eval
end

function compile_PM10_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 3/3)), 768)
    unit_size = Int(round(input_size * 3/3))
    @show "Unit size in PM10: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y)
    #loss(x, y) = huber_loss_mean(model(x), y)
    accuracy(data) = RSR(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_PM25_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 2/3)), 512)
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in PM25: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y)
    #loss(x, y) = huber_loss_mean(model(x), y)
    accuracy(data) = RSR(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_SO2_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 2/3)), 512)
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in SO2: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y)
    #loss(x, y) = huber_loss_mean(model(x), y)
    accuracy(data) = RSR(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end


function compile_NO2_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 2/3)), 512)
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in NO2: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y)
    #loss(x, y) = huber_loss_mean(model(x), y)
    accuracy(data) = RSR(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end


function compile_CO_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
    #unit_size = min(Int(round(input_size * 2/3)), 512)
    unit_size = Int(round(input_size * 2/3))
    @show "Unit size in CO: ", unit_size
    # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model = Chain(
        Dense(input_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, unit_size, leakyrelu), Dropout(0.2),

        Dense(unit_size, output_size)
    ) |> gpu

    loss(x, y) = Flux.mse(model(x), y)
    #loss(x, y) = huber_loss_mean(model(x), y)
    accuracy(data) = RSR(data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end