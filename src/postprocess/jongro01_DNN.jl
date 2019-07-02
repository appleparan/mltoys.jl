function findrow(df::DataFrame, col::Symbol, val)
    idx = 0
    for row in eachrow(df)
        idx += 1
        if (row[col] == val)
            return idx
        end
    end 

    idx = 0

    idx
end

function test_station(model_path::String, stn_df::DataFrame, ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights
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
    deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))

    test_wd_idxs = window_df(stn_df, sample_size, output_size, test_sdate, test_fdate)
    sidx = findrow(stn_df, :date, test_sdate)
    fidx = findrow(stn_df, :date, test_fdate)
    if (sidx == 0 || fidx == 0) 
        @error "Error; cannot found date: ", sidx, fidx
        return
    end

    test_idxs = sidx:fidx
    test_size = length(test_wd_idxs)

    μσs = mean_and_std_cols(stn_df, features)
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value

    test_μ, test_σ = mean_and_std(stn_df[test_idxs, norm_ycol])
    μσ = ndsparse((
        dataset = ["test", "test"],
        type = ["μ", "σ"]),
        (value = [test_μ, test_σ],))

    #@info "    Constructing (input, output) pairs for test set (another station)..."
    #p = Progress(length(test_wd_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    #test_set = [(ProgressMeter.next!(p); make_pairs_DNN(stn_df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]
    test_set = [(make_pairs_DNN(stn_df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]

    _RMSE = RMSE("test", test_set, model, μσ)
    _RSR = RSR("test", test_set, model, μσ)
    _PBIAS = PBIAS("test", test_set, model, μσ)
    _NSE = NSE("test", test_set, model, μσ)
    _IOA = IOA("test", test_set, model, μσ)

    test_dates = collect(test_sdate:Hour(1):test_fdate)
    table_01h, table_24h = compute_prediction(test_set, model, ycol, total_μ, total_σ, "/mnt/")
    ycol_str = string(ycol)
    y_01h_vals, ŷ_01h_vals, y_24h_vals, ŷ_24h_vals = export2CSV(DateTime.(test_dates), table_01h, table_24h, ycol, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h,
        DateTime(2018, 7, 1, 1), DateTime(2018, 7, 7, 23), ycol,
        output_dir, output_prefix)

    _corr_01h = Statistics.cor(y_01h_vals, ŷ_01h_vals)
    _corr_24h = Statistics.cor(y_24h_vals, ŷ_24h_vals)

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) RSR         : ", _RSR
    @info " $(string(ycol)) PBIAS       : ", _PBIAS
    @info " $(string(ycol)) NSE         : ", _NSE
    @info " $(string(ycol)) IOA         : ", _IOA
    
    @info " $(string(ycol)) Corr(01H)   : ", _corr_01h
    @info " $(string(ycol)) Corr(24H)   : ", _corr_24h

    stn_lat = stn_df[1, :lat]
    stn_lon = stn_df[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol),
        _RMSE, _RSR, _PBIAS, _NSE, _IOA, _corr_01h, _corr_24h]
end

function test_station(model_path::String, stn_df_in::DataFrame, stn_df_out::DataFrame, ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights
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

    test_idxs = sidx:fidx
    test_size = length(test_wd_idxs)

    μσs = mean_and_std_cols(stn_df_out, features)
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value

    test_μ, test_σ = mean_and_std(stn_df_out[test_idxs, norm_ycol])
    μσ = ndsparse((
        dataset = ["test", "test"],
        type = ["μ", "σ"]),
        (value = [test_μ, test_σ],))

    #@info "    Constructing (input, output) pairs for test set (another station)..."
    #p = Progress(length(test_wd_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    #test_set = [(ProgressMeter.next!(p); make_pairs_DNN(stn_df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]
    test_set = [(make_pairs_DNN(stn_df_in, stn_df_out, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]

    _RMSE = RMSE("test", test_set, model, μσ)
    _RSR = RSR("test", test_set, model, μσ)
    _PBIAS = PBIAS("test", test_set, model, μσ)
    _NSE = NSE("test", test_set, model, μσ)
    _IOA = IOA("test", test_set, model, μσ)

    test_dates = collect(test_sdate:Hour(1):test_fdate)
    table_01h, table_24h = compute_prediction(test_set, model, ycol, total_μ, total_σ, "/mnt/")
    ycol_str = string(ycol)
    y_01h_vals, ŷ_01h_vals, y_24h_vals, ŷ_24h_vals = export2CSV(DateTime.(test_dates), table_01h, table_24h, ycol, output_dir, output_prefix)

    plot_datefmt = @dateformat_str "yyyymmddHH"
    plot_DNN_lineplot(DateTime.(test_dates), table_01h, table_24h,
        DateTime(2018, 7, 1, 1), DateTime(2018, 7, 7, 23), ycol,
        output_dir, output_prefix)

    _corr_01h = Statistics.cor(y_01h_vals, ŷ_01h_vals)
    _corr_24h = Statistics.cor(y_24h_vals, ŷ_24h_vals)

    @info " $(string(ycol)) RMSE        : ", _RMSE
    @info " $(string(ycol)) RSR         : ", _RSR
    @info " $(string(ycol)) PBIAS       : ", _PBIAS
    @info " $(string(ycol)) NSE         : ", _NSE
    @info " $(string(ycol)) IOA         : ", _IOA

    @info " $(string(ycol)) Corr(01H)   : ", _corr_01h
    @info " $(string(ycol)) Corr(24H)   : ", _corr_24h

    stn_lat = stn_df_in[1, :lat]
    stn_lon = stn_df_in[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol),
        _RMSE, _RSR, _PBIAS, _NSE, _IOA, _corr_01h, _corr_24h]
end

function test_classification(model_path::String, stn_df::DataFrame, ycol::Symbol, stn_code::Integer, stn_name::String,
    sample_size::Integer, output_size::Integer,
    output_dir::String, output_prefix::String,
    test_sdate::ZonedDateTime=ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul"),
    test_fdate::ZonedDateTime=ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul"),
    is_zscored::Bool=false)

    filepath = model_path * string(ycol) * ".bson"

    BSON.@load filepath cpu_model weights
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
    deleteat!(norm_feas, findall(x -> x == norm_ycol, norm_feas))

    test_wd_idxs = window_df(stn_df, sample_size, output_size, test_sdate, test_fdate)
    sidx = findrow(stn_df, :date, test_sdate)
    fidx = findrow(stn_df, :date, test_fdate)
    if (sidx == 0 || fidx == 0) 
        @error "Error; cannot found date: ", sidx, fidx
        return
    end

    test_idxs = sidx:fidx
    test_size = length(test_wd_idxs)

    μσs = mean_and_std_cols(stn_df, features)
    total_μ = μσs[String(ycol), "μ"].value
    total_σ = μσs[String(ycol), "σ"].value

    test_μ, test_σ = mean_and_std(stn_df[test_idxs, norm_ycol])
    μσ = ndsparse((
        dataset = ["test", "test"],
        type = ["μ", "σ"]),
        (value = [test_μ, test_σ],))

    #@info "    Constructing (input, output) pairs for test set (another station)..."
    #p = Progress(length(test_wd_idxs), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    #test_set = [(ProgressMeter.next!(p); make_pairs_DNN(stn_df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]
    test_set = [(make_pairs_DNN(stn_df, norm_ycol, collect(idx), norm_feas, sample_size, output_size)) for idx in test_wd_idxs]

    forecast_all, forecast_high = classification("test", test_set, ycol, model)

    stn_lat = stn_df[1, :lat]
    stn_lon = stn_df[1, :lon]
    return [stn_code, stn_name, stn_lat, stn_lon, string(ycol), forecast_all, forecast_high]
end
