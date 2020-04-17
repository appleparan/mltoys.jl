using Random

using Dates, TimeZones
using StatsBase, Impute
using JuliaDB, DataFrames, DataFramesMeta
using Mise
using MicroLogging

function run_model()

    @info "Start Model"
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
    train_tdate = ZonedDateTime(2017, 12, 31, 23, tz"Asia/Seoul")
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
    train_stn_names = [:종로구, :강서구, :송파구, :강남구]
    train_stn_names = [:종로구]
    #=
    train_stn_names = [
        :중구,:종로구,:용산구,:광진구,:성동구,
        :중랑구,:동대문구,:성북구,:도봉구,:은평구,
        :서대문구,:마포구,:강서구,:구로구,:영등포구]
    =#

    # test set is Jongro only
    test_stn_names = [:종로구]

    df = filter_raw_data(rawdf, train_fdate, train_tdate, test_fdate, test_tdate)
    @show first(df, 10)

    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    # If you want exclude some features, modify train_features
    # exclude :PM10, :PM25 temporarily for log transform
    train_features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    target_features = [:PM10, :PM25]

    # For GPU, change precision of Floating numbers
    _eltype::DataType = Float32
    scaling_method = :zscore

    sample_size = 24
    output_size = 24
    epoch_size = 300
    batch_size = 32
    input_size = sample_size * length(train_features)
    #===== end of parameter zone =====#

    log_prefix = "log_"
    inv_prefix = "inv_"
    scaled_prefix = "scaled_"
    scaled_features = [Symbol(scaled_prefix, f) for f in features]
    scaled_train_features = [Symbol(scaled_prefix, f) for f in train_features]
    scaled_target_features = [Symbol(scaled_prefix, f) for f in target_features]

    # Preprocessing
    # 0. imputation
    DataFrames.allowmissing!(df, train_features)
    for ycol in train_features
        df[!, ycol] = Impute.srs(df[!, ycol])
    end
    DataFrames.disallowmissing!(df, train_features)

    # 2. Scaling
    if scaling_method == :zscore
        # scaling except :PM10, :PM25
        zscore!(df, train_features, scaled_train_features)

        statvals = extract_col_statvals(df, train_features)
    elseif scaling_method == :minmax
        minmax_scaling!(df, train_features, scaled_train_features, 0.0, 10.0)

        statvals = extract_col_statvals(df, train_features)
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

    @show first(df, 5)
    # convert Float types
    for feature in features
        df[!, feature] = _eltype.(df[!, feature])
    end

    # plot histogram regardless to station
    for target in target_features
        plot_histogram(df, target, "/mnt/")
        plot_histogram(df, Symbol(:scaled_, target), "/mnt/")

        if scaling_method == :logzscore || scaling_method == :logminmax
            plot_histogram(df, Symbol(:log_, target), "/mnt/")
        end

        if scaling_method == :invzscore
            plot_histogram(df, Symbol(:inv_, target), "/mnt/")
        end
    end

    # line plot of first train station
    plot_lineplot_total(filter_station(df, seoul_stations[train_stn_names[1]]), :PM10, "/mnt/")
    plot_lineplot_total(filter_station(df, seoul_stations[train_stn_names[1]]), :PM25, "/mnt/")
    #plot_pcorr(df, scaled_features, features, "/mnt/")

    # windowed dataframe for train/valid (input + output)
    test_wd = []

    # random permutation train_valid_wd itself for splitting train/valid set
    rng = MersenneTwister()

    # windows dataframe for test (only for 종로구)
    for name in test_stn_names
        code = seoul_stations[name]
        stn_df = filter_station_DNN(df, code)

        # to add zero(0) offset for X₀
        push!(test_wd,
            window_df(stn_df, sample_size, output_size + 1, test_fdate, test_tdate))
    end
    # Flatten
    test_wd = collect(Base.Iterators.flatten(test_wd))
    test_size = length(test_wd)

    # simply collect dates, determine exact date for prediction (for 1h, 24h, and so on) later
    test_dates = collect(test_fdate + Hour(sample_size - 1):Hour(1):test_tdate - Hour(output_size))

    for target in target_features
        @info "$(string(target)) Training..."
        flush(stdout); flush(stderr)

        @info "training feature : " train_features
        @info "sizes (sample, output, epoch, batch) : ", sample_size, output_size, epoch_size, batch_size

        for name in test_stn_names
            code = seoul_stations[name]
            stn_df = filter_station_DNN(df, code)

            Base.Filesystem.mkpath("/mnt/$(name)/")

            max_time = input_size * 15
            acf = StatsBase.autocor(stn_df[!, Symbol(scaled_prefix, target)], 0:max_time)

            # free minibatch after training because of memory usage
            ou_df = evolve_OU(test_wd, target, Symbol(scaled_prefix, target),
                acf, statvals, _eltype, sample_size, output_size, test_dates, "/mnt/$(name)/")

            # create directory per each time
            for i = 1:output_size
                i_pad = lpad(i, 2, '0')
                Base.Filesystem.mkpath("/mnt/$(name)/$(i_pad)/")
            end

            dfs_out = export_CSV(DateTime.(test_dates, Local), ou_df, target, output_size, "/mnt/$(name)/", String(target))
            plot_DNN_scatter(dfs_out, target, output_size, "/mnt/$(name)/", String(target))
            plot_DNN_histogram(dfs_out, target, output_size, "/mnt/$(name)/", String(target))

            plot_DNN_lineplot(dfs_out, target, output_size, "/mnt/$(name)/", String(target))

            df_corr = compute_corr(dfs_out, output_size, "/mnt/$(name)/", String(target))
            plot_corr(df_corr, output_size, "/mnt/$(name)/", String(target))
        end
    end
end

run_model()
