using Random

using Dates, TimeZones
using StatsBase, Impute
using JuliaDB, DataFrames, DataFramesMeta
using Mise
using MicroLogging

function run_model()

    @info "Start ARIMA"
    flush(stdout); flush(stderr)

    seoul_codes = [
        111121,111123,111131,111141,111142,
        111151,111152,111161,111171,111181,
        111191,111201,111212,111221,111231,
        111241,111251,111261,111262,111273,
        111274,111281,111291,111291,111301,
        111301,111311]
    seoul_names = [
        "중구","종로구","용산구","광진구","성동구",
        "중랑구","동대문구","성북구","도봉구","은평구",
        "서대문구","마포구","강서구","구로구","영등포구",
        "동작구","관악구","강남구","서초구","송파구",
        "강동구","금천구","강북구","강북구","양천구",
        "양천구","노원구"]
    # construct named tuple
    seoul_stations = (; zip(Symbol.(seoul_names), seoul_codes)...)

    # NamedTuple (::Symbol -> ::DataFrame)
    rawdf = load_data_DNN("/input/jongro_seoul.csv", seoul_stations)
    #=
    first(df, 5)
    │ Row │ stationCode │ date                      │ lat      │ long      │ SO2     │ CO      │ O3      │ NO2     │ PM10    │ PM25    │ temp    │ u       │ v       │ pres    │ prep     │ snow     │
    │     │ Int64       │ TimeZones.ZonedDateTime   │ Float64  │ Float64   │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64  │
    ├─────┼─────────────┼───────────────────────────┼──────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤─────────┤──────────┤──────────┤
    │ 1   │ 111123      │ 2015-01-01T01:00:00+09:00 │ 37.572   │ 127.005   │ 0.004   │ 0.2     │ 0.02    │ 0.009   │ 57.0    │ 0.0     │ -7.4    │ 4.41656 │ 1.60749 │ 1011.8  │ missing  │ missing  │
    │ 2   │ 111123      │ 2015-01-01T02:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.019   │ 0.008   │ 70.0    │ 3.0     │ -8.0    │ 4.22862 │ 1.53909 │ 1011.7  │ missing  │ missing  │
    │ 3   │ 111123      │ 2015-01-01T03:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.02    │ 0.006   │ 92.0    │ 5.0     │ -8.4    │ 3.57083 │ 1.29968 │ 1012.1  │ missing  │ missing  │
    │ 4   │ 111123      │ 2015-01-01T04:00:00+09:00 │ 37.572   │ 127.005   │ 0.004   │ 0.2     │ 0.019   │ 0.005   │ 111.0   │ 2.0     │ -8.8    │ 4.60449 │ 1.6759  │ 1012.3  │ missing  │ missing  │
    │ 5   │ 111123      │ 2015-01-01T05:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.019   │ 0.006   │ 127.0   │ 5.0     │ -9.1    │ 5.35625 │ 1.94951 │ 1011.8  │ missing  │ missing  │
    =#

    #===== start of parameter zone =====#
    total_fdate, total_tdate = get_date_range(rawdf)
    train_fdate = ZonedDateTime(2012, 1, 1, 0, tz"Asia/Seoul")
    train_tdate = ZonedDateTime(2017, 12, 31, 24, tz"Asia/Seoul")
    test_fdate = ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul")
    test_tdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

    # due to seasonality, train date range must be over 1 year
    @assert train_tdate - train_fdate >= Dates.Day(365)

    # stations
    #=
    train_stn_names = [
        :중구,:종로구,:용산구,:광진구,:성동구,
        :중랑구,:동대문구,:성북구,:도봉구,:은평구,
        :서대문구,:마포구,:강서구,:구로구,:영등포구,
        :동작구,:관악구,:강남구,:서초구,:송파구,
        :강동구,:금천구,:강북구,:강북구,:양천구,
        :양천구,:노원구]
    =#

    # ARIMA : univariate & single station
    train_stn_names = [:종로구]
    test_stn_names = train_stn_names

    df = filter_raw_data(rawdf, train_fdate, train_tdate, test_fdate, test_tdate)
    @show first(df, 10)

    features = [:PM10, :PM25]
    # If you want exclude some features, modify train_features

    # Univariate Prediction
    target_features = [:PM10, :PM25]
    train_features = target_features

    # For GPU, change precision of Floating numbers
    _eltype::DataType = Float32
    scaling_method = :zscore

    sample_size = 24
    output_size = 24
    epoch_size = 300
    batch_size = 32
    input_size = 24*365 # 1-year
    #===== end of parameter zone =====#

    log_prefix = "log_"
    inv_prefix = "inv_"
    scaled_prefix = "scaled_"
    scaled_features = [Symbol(scaled_prefix, f) for f in features]
    scaled_train_features = [Symbol(scaled_prefix, f) for f in train_features]
    scaled_target_features = [Symbol(scaled_prefix, f) for f in target_features]

    # to train residuals
    train_features_res = copy(train_features)
    target_features_res = copy(target_features)
    replace!(train_features_res, :PM10 => :PM10_res, :PM25 => :PM25_res)
    replace!(target_features_res, :PM10 => :PM10_res, :PM25 => :PM25_res)
    scaled_train_features_res = [Symbol(scaled_prefix, f) for f in train_features_res]
    scaled_target_features_res = [Symbol(scaled_prefix, f) for f in target_features_res]

    # Preprocessing
    # 0. imputation
    DataFrames.allowmissing!(df, train_features)
    for ycol in train_features
        df[!, ycol] = Impute.srs(df[!, ycol])
    end
    DataFrames.disallowmissing!(df, train_features)

    season_tables = Dict{Symbol, AbstractNDSparse}()
    # 1. Decompose seasonality of particulate matter by total station
    for target in target_features
        # this assumes seasonality only resctricts to particulate matter, also PM has annual and daily seasonality
        lee_year_sea1, lee_year_sea2, lee_year_res2, lee_day_sea1, lee_day_res1 =
            season_adj_lee(df, Symbol(target), train_fdate, train_tdate)

        # append seasonality to df for train_date
        # smoothed annual seasonality
        padded_push!(df, lee_year_sea1, Symbol(target, "_year_org"), train_fdate, train_tdate)
        padded_push!(df, lee_year_sea2, Symbol(target, "_year_fit"), train_fdate, train_tdate)
        # daily seasonality
        padded_push!(df, lee_day_sea1, Symbol(target, "_day"), train_fdate, train_tdate)
        # residual
        #padded_push!(df, lee_day_res1, Symbol(target, "_res"), train_fdate, train_tdate)
        res_arr = df[!, target] .- df[!, Symbol(target, "_year_fit")] .- df[!, Symbol(target, "_day")]
        DataFrames.insertcols!(df, 3, Symbol(target, "_res") => res_arr)

        season_table = construct_annual_table(df, target, DateTime(train_fdate, Local), DateTime(train_tdate, Local))
        season_tables[target] = season_table

        # add seasonality to df (existing column) for test_date
        # annual seasonality is a smoothed version
        padded_push!(df, target,
            Symbol(target, "_year_fit"),
            Symbol(target, "_year_org"),
            Symbol(target, "_day"),
            Symbol(target, "_res"),
            season_table, test_fdate, test_tdate)
    end

    # 2. Scaling
    if scaling_method == :zscore
        # scaling except :PM10, :PM25
        zscore!(df, train_features_res, scaled_train_features_res)

        statvals = extract_col_statvals(df, train_features_res)
    elseif scaling_method == :minmax
        minmax_scaling!(df, train_features, scaled_train_features, 0.0, 10.0)

        statvals = extract_col_statvals(df, train_features_res)
    elseif scaling_method == :logzscore
        # zscore except targets
        _train_features = setdiff(train_features, target_features)
        _scaled_train_features = setdiff(scaled_train_features, scaled_target_features)
        zscore!(df, _train_features, _scaled_train_features)

        # Log transform :PM10, :PM25
        # add 10.0 not to log zero values
        # Y = zscore(log(X + 10))
        log_target_features = [Symbol(log_prefix, f) for f in target_features]
        for (target, log_target) in zip(target_features, log_target_features)
            df[!, log_target] = log.(df[!, target] .+ 10.0)
        end

        zscore!(df, log_target_features, scaled_target_features)

        statvals = extract_col_statvals(df, vcat(train_features, log_target_features))
    elseif scaling_method == :invzscore
        # zscore except targets
        _train_features = setdiff(train_features, target_features)
        _scaled_train_features = setdiff(scaled_train_features, scaled_target_features)
        zscore!(df, _train_features, _scaled_train_features)

        # Inverse transform :PM10, :PM25
        # add 10.0 not to divide zero value
        # Y = zscore(1.0 / (X + 10))
        inv_target_features = [Symbol(inv_prefix, f) for f in target_features]
        for (target, inv_target) in zip(target_features, inv_target_features)
            df[!, inv_target] = 1.0 ./ (df[!, target] .+ 10.0)
        end

        zscore!(df, inv_target_features, scaled_target_features)

        statvals = extract_col_statvals(df, vcat(train_features, inv_target_features))
    elseif scaling_method == :logminmax
        # minmax except targets
        _train_features = setdiff(train_features, target_features)
        _scaled_train_features = setdiff(scaled_train_features, scaled_target_features)
        minmax_scaling!(df, _train_features, _scaled_train_features, 0.0, 10.0)

        # Log transform :PM10, :PM25
        # add 10.0 not to log zero values
        log_target_features = [Symbol(log_prefix, f) for f in target_features]
        for (target, log_target) in zip(target_features, log_target_features)
            df[!, log_target] = log.(df[!, target] .+ 10.0)
        end

        # then minmax
        minmax_scaling!(df, log_target_features, scaled_target_features, 0.0, 10.0)

        statvals = extract_col_statvals(df, vcat(train_features, log_target_features))
    end

    # convert Float types
    for feature in features
        df[!, feature] = _eltype.(df[!, feature])
    end

    # windowed dataframe for train/valid (input + output)
    train_dfs = Dict()
    test_dfs = Dict()

    for name in train_stn_names
        code = seoul_stations[name]
        _stn_df = filter_station_DNN(df, code)

        # get long dataframe
        test_df = DataFrames.filter(row -> test_fdate - Dates.Hour(input_size) <= row[:date] <= test_tdate, _stn_df)

        test_dfs[name] = test_df
    end

    for target in target_features
        @info "$(string(target)) Training..."
        flush(stdout); flush(stderr)

        @info "training feature : " train_features
        @info "sizes (input, output, epoch, batch) : ", input_size, output_size, epoch_size, batch_size

        # add seasonality
        for name in test_stn_names
            code = seoul_stations[name]

            Base.Filesystem.mkpath("/mnt/$(name)/")

            @show first(test_dfs[name], 5)

            max_time = 24 * 15
            acf_dres1 = StatsBase.autocor(test_dfs[name][!, Symbol(scaled_prefix, target, "_res")], 0:max_time)
            pacf_dres1 = StatsBase.pacf(test_dfs[name][!, Symbol(scaled_prefix, target, "_res")], 0:max_time)

            # free minibatch after training because of memory usage
            arima_df = predict_ARIMA(test_dfs[name], target, Symbol(scaled_prefix, target, "_res"),
                acf_dres1, pacf_dres1, season_tables[target], statvals,
                train_fdate, train_tdate, test_fdate, test_tdate,
                 _eltype, input_size, output_size, "/mnt/$(name)/", :ARIMA)

            # create directory per each time
            for i = 1:output_size
                i_pad = lpad(i, 2, '0')
                Base.Filesystem.mkpath("/mnt/$(name)/$(i_pad)/")
            end

            dfs_out = export_CSV(arima_df[:, :date], arima_df, season_tables[target], target, output_size, "/mnt/$(name)/", String(target))
            plot_ARIMA_scatter(dfs_out, target, output_size, "/mnt/$(name)/", String(target), :ARIMA)
            plot_ARIMA_lineplot(dfs_out, target, output_size, "/mnt/$(name)/", String(target), :ARIMA)

            df_corr = compute_corr(dfs_out, output_size, "/mnt/$(name)/", String(target))
            plot_ARIMA_corr(df_corr, output_size, "/mnt/$(name)/", String(target), :ARIMA)
        end
    end
end

run_model()
