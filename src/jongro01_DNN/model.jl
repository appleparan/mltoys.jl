function train_all_DNN(df::DataFrame, norm_feas::Array{Symbol}, norm_prefix::String,
    sample_size::Integer, input_size::Integer, batch_size::Integer, output_size::Integer, epoch_size::Integer,
    total_wd_idxs::Array{Any, 1}, test_wd_idxs::Array{Any, 1}, train_chnk::Array{T, 1}, valid_idxs::Array{I, 1}, test_idxs::Array{I, 1},
    μσs::AbstractNDSparse, test_dates::Array{ZonedDateTime,1}) where T <: Array{I, 1} where I <: Integer

    @info "PM10 Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    PM10_model, PM10_μσ = train_DNN(df, :PM10, norm_prefix, norm_feas,
    sample_size, input_size, batch_size, output_size, epoch_size,
    total_wd_idxs, test_wd_idxs, train_chnk, valid_idxs, test_idxs, μσs,
    "PM10", test_dates)

    @info "PM25 Training..."
    flush(stdout); flush(stderr)

    PM25_model, PM25_μσ = train_DNN(df, :PM25, norm_prefix, norm_feas,
    sample_size, input_size, batch_size, output_size, epoch_size,
    total_wd_idxs, test_wd_idxs, train_chnk, valid_idxs, test_idxs, μσs,
    "PM25", test_dates)

    nothing
end

"""
    train(df, ycol, norm_prefix, norm_feas,
        sample_size, input_size, batch_size, output_size, epoch_size,
        total_wd_idxs, test_wd_idxs,
        train_chnk, valid_idxs, test_idxs,
        μσs, filename, test_dates)

"""
function train_DNN(df::DataFrame, ycol::Symbol, norm_prefix::String, _norm_feas::Array{Symbol},
    sample_size::Integer, input_size::Integer, batch_size::Integer, output_size::Integer, epoch_size::Integer,
    total_wd_idxs::Array{Any, 1}, test_wd_idxs::Array{Any, 1}, 
    train_chnk::Array{T, 1}, valid_idxs::Array{I, 1}, test_idxs::Array{I, 1},
    μσs::AbstractNDSparse, filename::String, test_dates::Array{ZonedDateTime,1}) where T <: Array{I, 1} where I <: Integer

    @info "DNN training starts.."

    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))

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
    compile = eval(Symbol(:compile, "_", ycol, "_DNN"))
    model, loss, accuracy, opt = compile(input_size, batch_size, output_size, μσ)

    # create (input(1D), output(1D)) pairs of total dataframe row, it is indepdent by train/valid/test set
    @info "    Constructing (input, output) pairs for train/valid set..."
    p = Progress(length(total_wd_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    input_pairs = [(ProgressMeter.next!(p); make_pairs_DNN(df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in total_wd_idxs]

    @info "    Constructing (input, output) pairs for test set..."
    p = Progress(length(test_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p); make_pairs_DNN(df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]

    @info "    Removing sparse datas..."
    p = Progress(length(input_pairs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    remove_missing_pairs!(input_pairs, 0.5, p)

    # |> gpu doesn't work to *_set directly
    # construct minibatch for train_set
    # https://github.com/FluxML/Flux.jl/issues/704    
    @info "    Constructing minibatch..."
    p = Progress(length(train_chnk), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    # total_wd_idxs[chnk] = get list of pair indexes -> i.e. [1, 2, 3, 4]
    train_set = [(ProgressMeter.next!(p); make_minibatch_DNN(input_pairs, chnk, batch_size)) for chnk in train_chnk]

    # don't construct minibatch for valid & test sets
    valid_set = input_pairs[valid_idxs]

    df_eval = train_DNN!(model, train_set, valid_set, loss, accuracy, opt, epoch_size, μσ, filename)

    # TODO : (current) validation with zscore, (future) validation with original value?
    @info "    Valid ACC : ", accuracy("valid", valid_set)
    @info "    Test ACC  : ", accuracy("test", test_set)
    flush(stdout); flush(stderr)

    @info " $(string(ycol)) RMSE for test   : ", RMSE("test", test_set, model, μσ)
    @info " $(string(ycol)) RSR for test    : ", RSR("test", test_set, model, μσ)
    @info " $(string(ycol)) PBIAS for test  : ", PBIAS("test", test_set, model, μσ)
    @info " $(string(ycol)) NSE for test    : ", NSE("test", test_set, model, μσ)
    @info " $(string(ycol)) IOA for test    : ", IOA("test", test_set, model, μσ)

    @info " $(string(ycol)) RMSE for valid  : ", RMSE("valid", valid_set, model, μσ)
    @info " $(string(ycol)) RSR for valid   : ", RSR("valid", valid_set, model, μσ)
    @info " $(string(ycol)) PBIAS for valid : ", PBIAS("valid", valid_set, model, μσ)
    @info " $(string(ycol)) NSE for valid   : ", NSE("valid", valid_set, model, μσ)
    @info " $(string(ycol)) IOA for valid   : ", IOA("valid", valid_set, model, μσ)

    forecast_all, forecast_high = classification("test", test_set, ycol, model)
    @info " $(string(ycol)) Forecasting accuracy (all) for test : ", forecast_all
    @info " $(string(ycol)) Forecasting accuracy (high) for test : ", forecast_high

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

    #=
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h, 
        DateTime(2018, 1, 4, 1), DateTime(2018, 3, 31, 23), ycol,
        "/mnt/", string(ycol) * "_" * Dates.format(DateTime(2018, 1, 4, 1), plot_datefmt) * "_" * Dates.format(DateTime(2018, 3, 31, 1), plot_datefmt))
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h,
        DateTime(2018, 4, 1, 1), DateTime(2018, 6, 30, 23), ycol,
        "/mnt/", string(ycol) * "_" * Dates.format(DateTime(2018, 4, 1, 1), plot_datefmt) * "_" * Dates.format(DateTime(2018, 6, 30, 1), plot_datefmt))
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h,
        DateTime(2018, 7, 1, 1), DateTime(2018, 9, 30, 23), ycol,
        "/mnt/", string(ycol) * "_" * Dates.format(DateTime(2018, 7, 1, 1), plot_datefmt) * "_" * Dates.format(DateTime(2018, 9, 30, 1), plot_datefmt))
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h,
        DateTime(2018, 10, 1, 1), DateTime(2018, 12, 27, 23), ycol,
        "/mnt/", string(ycol) * "_" * Dates.format(DateTime(2018, 10, 1, 1), plot_datefmt) * "_" * Dates.format(DateTime(2018, 12, 31, 1), plot_datefmt))
    =#

    plot_evaluation(df_eval, ycol, "/mnt/")

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
        rmse, rsr, nse, pbias, ioa = evaluations("valid", valid_set, model, μσ, [:RMSE, :RSR, :NSE, :PBIAS, :IOA])
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
            # TrackedReal cannot be writable, convert to Real
            filepath = "/mnt/" * filename * ".bson"
            BSON.@save filepath cpu_model epoch_idx acc
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
    accuracy(setname, data) = RSR(setname, data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end

function compile_PM25_DNN(input_size::Integer, batch_size::Integer, output_size::Integer, μσ)
    @info("    Compiling model...")
    # answer from SO: https://stats.stackexchange.com/a/180052
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
    accuracy(setname, data) = RSR(setname, data, model, μσ)
    opt = Flux.ADAM()

    model, loss, accuracy, opt
end
