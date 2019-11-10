"""
test_station(model_path, stn_df, ycol, stn_code, stn_name,
    sample_size, output_size,
    output_dir, output_prefix;
    test_fdate=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_tdate==ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    eltype=Float32,
    is_zscored=false)

Dirty codes for post processing
"""
function test_station(model_path::String, stn_df::DataFrame,
    ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_tdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    eltype::DataType=Float32, is_normalize::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # norm_features and features
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    if (is_normalize == false)
        min_max_scaling!(stn_df, features, _norm_feas)
    end
    
    # type conversion
    for fea in features
        stn_df[!, fea] = eltype.(stn_df[!, fea])
    end

    for nfea in _norm_feas
        stn_df[!, nfea] = eltype.(stn_df[!, nfea])
    end

    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd = window_df(stn_df, sample_size, output_size, test_fdate, test_tdate)
    test_wd_size = length(test_wd)

    μσs = mean_and_std_cols(stn_df, features)
    stn_μ = μσs[String(ycol), "μ"].value
    stn_σ = μσs[String(ycol), "σ"].value
    total_min = float(μσs[String(ycol), "minimum"].value)
    total_max = float(μσs[String(ycol), "maximum"].value)
    
    μσ = ndsparse((
        dataset = ["total", "total"],
        type = ["μ", "σ"]),
        (value = [μ, σ],))

    @info "    Construct Test Set..."
    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p);
        make_pair_DNN(df, norm_ycol, norm_feas, sample_size, output_size, eltype)) for df in test_wd]

    _RMSE = RMSE(test_set, model, μσ)
    _MAE = MAE(test_set, model, μσ)
    _MSPE = MSPE(test_set, model, μσ)
    _MAPE = MAPE(test_set, model, μσ)

    #test_dates = collect(test_fdate:Hour(1):test_tdate)
    test_dates = collect(test_fdate + Hour(sample_size - 1):Hour(1):test_tdate - Hour(output_size))

    #dnn_table = predict_model(test_set, model, ycol, μ, σ, output_size, output_dir)
    dnn_table = predict_model_minmax(test_set, model, ycol, total_min, total_max,
        output_size, "/mnt/")
    dfs_out = export_CSV(DateTime.(test_dates), dnn_table, ycol, output_size, output_dir, output_prefix)
    df_corr = compute_corr(dnn_table, output_size, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    test_fdate_str = Dates.format(test_fdate, plot_datefmt)
    test_tdate_str = Dates.format(test_tdate, plot_datefmt)
    plot_DNN_lineplot(DateTime.(test_dates), dnn_table,
        DateTime(test_fdate), DateTime(test_tdate), ycol, output_size, output_dir,
        output_prefix * "_$(test_fdate_str)_$(test_tdate_str)")

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) MAE         : ", _MAE
    @info " $(string(ycol)) MSPE        : ", _MSPE
    @info " $(string(ycol)) MAPE        : ", _MAPE

    stn_lat = stn_df[1, :lat]
    stn_lon = stn_df[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol), _RMSE, _MAE, _MSPE, _MAPE]
end

function test_station(model_path::String, stn_df_in::DataFrame, stn_df_out::DataFrame,
    ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_tdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    eltype::DataType=Float32, is_normalize::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # create dataset
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    if (is_normalize == false)
        #zscore!(stn_df_in, features, _norm_feas)
        min_max_scaling!(stn_df_in, features, _norm_feas)
    end
    
    # type conversion
    for fea in features
        stn_df_in[!, fea] = eltype.(stn_df_in[!, fea])
    end

    for nfea in _norm_feas
        stn_df_in[!, nfea] = eltype.(stn_df_in[!, nfea])
    end
    
    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd = window_df(stn_df_in, sample_size, output_size, test_fdate, test_tdate)
    test_wd_size = length(test_wd)

    μσs = mean_and_std_cols(stn_df_out, features)
    stn_μ = μσs[String(ycol), "μ"].value
    stn_σ = μσs[String(ycol), "σ"].value
    total_min = float(μσs[String(ycol), "minimum"].value)
    total_max = float(μσs[String(ycol), "maximum"].value)
    
    μσ = ndsparse((
        dataset = ["total", "total"],
        type = ["μ", "σ"]),
        (value = [μ, σ],))

    p = Progress(length(test_wd), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    test_set = [(ProgressMeter.next!(p);
        @show length(df), output_size;
        make_pair_DNN(df, norm_ycol, norm_feas, sample_size, output_size, eltype)) for df in test_wd]

    _RMSE = RMSE(test_set, model, μσ)
    _MAE = MAE(test_set, model, μσ)
    _MSPE = MSPE(test_set, model, μσ)
    _MAPE = MAPE(test_set, model, μσ)

    #test_dates = collect(test_fdate:Hour(1):test_tdate)
    test_dates = collect(test_fdate + Hour(sample_size - 1):Hour(1):test_tdate - Hour(output_size))

    #dnn_table = predict_model(test_set, model, ycol, μ, σ, output_size, output_dir)
    dnn_table = predict_model_minmax(test_set, model, ycol, total_min, total_max,
        output_size, "/mnt/")
    dfs_out = export_CSV(DateTime.(test_dates), dnn_table, ycol, output_size, output_dir, output_prefix)
    df_corr = compute_corr(dnn_table, output_size, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    test_fdate_str = Dates.format(test_fdate, plot_datefmt)
    test_tdate_str = Dates.format(test_tdate, plot_datefmt)
    plot_DNN_lineplot(DateTime.(test_dates), dnn_table,
        DateTime(test_fdate), DateTime(test_tdate), ycol, output_size, output_dir,
        output_prefix * "_$(test_fdate_str)_$(test_tdate_str)")

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) MAE         : ", _MAE
    @info " $(string(ycol)) MSPE        : ", _MSPE
    @info " $(string(ycol)) MAPE        : ", _MAPE

    stn_lat = stn_df_in[1, :lat]
    stn_lon = stn_df_in[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol), _RMSE, _MAE, _MSPE, _MAPE]
end

function test_classification(model_path::String, stn_df::DataFrame, ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_tdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    eltype::DataType=Float32, is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # create dataset
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    # type conversion
    for fea in features
        stn_df[!, fea] = eltype.(stn_df[!, fea])
    end

    for nfea in _norm_feas
        stn_df[!, nfea] = eltype.(stn_df[!, nfea])
    end

    if (is_zscored == false)
        zscore!(stn_df, features, _norm_feas)
    end
    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd_idxs = window_df(stn_df, sample_size, output_size, test_fdate, test_tdate)
    sidx = findrow(stn_df, :date, test_fdate)
    fidx = findrow(stn_df, :date, test_tdate)
    if (sidx == 0 || fidx == 0) 
        @error "Error; cannot found date: ", sidx, fidx
        return
    end
    test_wd_size = length(test_wd_idxs)
    test_size = length(test_wd_idxs)
    test_idxs = create_idxs(test_wd_idxs, test_size)

    μσs = mean_and_std_cols(stn_df, features)
    stn_μ = μσs[String(ycol), "μ"].value
    stn_σ = μσs[String(ycol), "σ"].value
    
    μσ = ndsparse((
        dataset = ["total", "total"],
        type = ["μ", "σ"]),
        (value = [μ, σ],))

    @info "    Construct Test Set..."
    test_set = [make_pair_DNN(stn_df, norm_ycol, idx, norm_feas, sample_size, output_size) for idx in test_wd_idxs]

    forecast_all, forecast_high = classification(test_set, model, ycol)

    stn_lat = stn_df[1, :lat]
    stn_lon = stn_df[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol), forecast_all, forecast_high]
end
