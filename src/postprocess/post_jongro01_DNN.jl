function test_station(model_path::String, stn_df::DataFrame,
    ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # create dataset
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    if (is_zscored == false)
        zscore!(stn_df, features, _norm_feas)
    end

    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd_idxs = window_df(stn_df, sample_size, output_size, test_sdate, test_fdate)
    sidx = findrow(stn_df, :date, test_sdate)
    fidx = findrow(stn_df, :date, test_fdate)
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

    _RMSE = RMSE(test_set, model, μσ)
    _RSR = RSR(test_set, model, μσ)
    _PBIAS = PBIAS(test_set, model, μσ)
    _NSE = NSE(test_set, model, μσ)
    _IOA = IOA(test_set, model, μσ)

    #test_dates = collect(test_sdate:Hour(1):test_fdate)
    test_dates = collect(test_sdate + Hour(sample_size - 1):Hour(1):test_fdate - Hour(output_size))

    dnn_table = predict_model(test_set, model, ycol, μ, σ, output_size, output_dir)
    dfs_out = export_CSV(DateTime.(test_dates), dnn_table, ycol, output_size, output_dir, output_prefix)
    df_corr = compute_corr(dnn_table, output_size, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    test_sdate_str = Dates.format(test_sdate, plot_datefmt)
    test_fdate_str = Dates.format(test_fdate, plot_datefmt)
    plot_DNN_lineplot(DateTime.(test_dates), dnn_table,
        DateTime(test_sdate), DateTime(test_fdate), ycol, output_size, output_dir,
        output_prefix * "_$(test_sdate_str)_$(test_fdate_str)")

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) RSR         : ", _RSR
    @info " $(string(ycol)) PBIAS       : ", _PBIAS
    @info " $(string(ycol)) NSE         : ", _NSE
    @info " $(string(ycol)) IOA         : ", _IOA

    stn_lat = stn_df[1, :lat]
    stn_lon = stn_df[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol),
        _RMSE, _RSR, _PBIAS, _NSE, _IOA]
end

function test_station(model_path::String, stn_df_in::DataFrame, stn_df_out::DataFrame,
    ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # create dataset
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    if (is_zscored == false)
        zscore!(stn_df_in, features, _norm_feas)
    end
    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd_idxs = window_df(stn_df_in, sample_size, output_size, test_sdate, test_fdate)
    sidx = findrow(stn_df_in, :date, test_sdate)
    fidx = findrow(stn_df_in, :date, test_fdate)
    if (sidx == 0 || fidx == 0) 
        @error "Error; cannot found date: ", sidx, fidx
        return
    end
    test_wd_size = length(test_wd_idxs)
    test_size = length(test_wd_idxs)
    test_idxs = create_idxs(test_wd_idxs, test_size)

    μσs = mean_and_std_cols(stn_df_out, features)
    stn_μ = μσs[String(ycol), "μ"].value
    stn_σ = μσs[String(ycol), "σ"].value
    
    μσ = ndsparse((
        dataset = ["total", "total"],
        type = ["μ", "σ"]),
        (value = [μ, σ],))

    @info "    Construct Test Set..."
    test_set = [make_pair_DNN(stn_df_in, norm_ycol, idx, norm_feas, sample_size, output_size) for idx in test_wd_idxs]

    _RMSE = RMSE(test_set, model, μσ)
    _RSR = RSR(test_set, model, μσ)
    _PBIAS = PBIAS(test_set, model, μσ)
    _NSE = NSE(test_set, model, μσ)
    _IOA = IOA(test_set, model, μσ)

    #test_dates = collect(test_sdate:Hour(1):test_fdate)
    test_dates = collect(test_sdate + Hour(sample_size - 1):Hour(1):test_fdate - Hour(output_size))

    dnn_table = predict_model(test_set, model, ycol, μ, σ, output_size, output_dir)
    dfs_out = export_CSV(DateTime.(test_dates), dnn_table, ycol, output_size, output_dir, output_prefix)
    df_corr = compute_corr(dnn_table, output_size, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    test_sdate_str = Dates.format(test_sdate, plot_datefmt)
    test_fdate_str = Dates.format(test_fdate, plot_datefmt)
    plot_DNN_lineplot(DateTime.(test_dates), dnn_table,
        DateTime(test_sdate), DateTime(test_fdate), ycol, output_size, output_dir,
        output_prefix * "_$(test_sdate_str)_$(test_fdate_str)")

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) RSR         : ", _RSR
    @info " $(string(ycol)) PBIAS       : ", _PBIAS
    @info " $(string(ycol)) NSE         : ", _NSE
    @info " $(string(ycol)) IOA         : ", _IOA

    stn_lat = stn_df_in[1, :lat]
    stn_lon = stn_df_in[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol),
        _RMSE, _RSR, _PBIAS, _NSE, _IOA]
end

function test_classification(model_path::String, stn_df::DataFrame, ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights μ σ
    Flux.loadparams!(cpu_model, weights)
    model = cpu_model |> gpu

    # create dataset
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]

    if (is_zscored == false)
        zscore!(stn_df, features, _norm_feas)
    end
    # remove target (ycol)
    norm_ycol = Symbol(norm_prefix, ycol)
    norm_feas = copy(_norm_feas)
    # remove ycol itself
    # deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))
    test_wd_idxs = window_df(stn_df, sample_size, output_size, test_sdate, test_fdate)
    sidx = findrow(stn_df, :date, test_sdate)
    fidx = findrow(stn_df, :date, test_fdate)
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
