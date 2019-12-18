using Random

using Dates, TimeZones
using MicroLogging
using StatsBase
using Mise

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
    df = load_data_DNN("/input/jongro_seoul.csv", seoul_stations)
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
    total_fdate, total_tdate = get_date_range(df)
    train_fdate = ZonedDateTime(2012, 1, 1, 1, tz"Asia/Seoul")
    train_tdate = ZonedDateTime(2017, 12, 31, 23, tz"Asia/Seoul")
    test_fdate = ZonedDateTime(2018, 1, 1, 0, tz"Asia/Seoul")
    test_tdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

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
    #train_stn_names = [:종로구]

    # test set is Jongro only
    test_stn_names = [:종로구]
    
    df = filter_raw_data(df, train_fdate, train_tdate, test_fdate, test_tdate)
    @show first(df, 10)

    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    # If you want exclude some features, modify train_features
    # exclude :PM10, :PM25 temporarily for log transform
    train_features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    target_features = [:PM10, :PM25]

    # For GPU, change precision of Floating numbers
    eltype::DataType = Float32
    scaling_method = :logzscore

    sample_size = 24
    output_size = 24
    epoch_size = 300
    batch_size = 32
    input_size = sample_size * length(train_features)
    #===== end of parameter zone =====#

    log_prefix = "log_"
    scaled_prefix = "scaled_"
    scaled_features = [Symbol(scaled_prefix, f) for f in features]
    scaled_train_features = [Symbol(scaled_prefix, f) for f in train_features]
    scaled_target_features = [Symbol(scaled_prefix, f) for f in target_features]

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
        log_target_features = [Symbol(log_prefix, f) for f in target_features]
        for (target, log_target) in zip(target_features, log_target_features)
            df[!, log_target] = log.(df[!, target] .+ 10.0)
        end

        # then zscore targets
        zscore!(df, log_target_features, scaled_target_features)

        statvals = extract_col_statvals(df, vcat(train_features, log_target_features))
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

        ## then zscore
        minmax_scaling!(df, log_target_features, scaled_target_features, 0.0, 10.0)

        statvals = extract_col_statvals(df, vcat(train_features, log_target_features))
    end

    # convert Float types
    for fea in features
        df[!, fea] = eltype.(df[!, fea])
    end

    for nfea in scaled_features
        df[!, nfea] = eltype.(df[!, nfea])
    end

    # plot histogram regardless to station
    plot_histogram(df, :PM25, "/mnt/")
    plot_histogram(df, :PM10, "/mnt/")
    plot_histogram(df, :scaled_PM25, "/mnt/")
    plot_histogram(df, :scaled_PM10, "/mnt/")

    if scaling_method == :logzscore || scaling_method == :logminmax
        plot_histogram(df, :log_PM25, "/mnt/")
        plot_histogram(df, :log_PM10, "/mnt/")
    end

    # line plot of first train station
    plot_lineplot_total(filter_station(df, seoul_stations[train_stn_names[1]]), :PM10, "/mnt/")
    plot_lineplot_total(filter_station(df, seoul_stations[train_stn_names[1]]), :PM25, "/mnt/")
    #plot_pcorr(df, scaled_features, features, "/mnt/")

    # windowed dataframe for train/valid (input + output)
    train_valid_wd = []
    test_wd = []

    # DNN's windowed dataframe is already filterred with dates.
    # size(df) = (sample_size, length(features))
    for name in train_stn_names
        code = seoul_stations[name]
        stn_df = filter_station_DNN(df, code)

        push!(train_valid_wd,
            window_df(stn_df, sample_size, output_size, train_fdate, train_tdate))
    end

    # Flatten 
    train_valid_wd = collect(Base.Iterators.flatten(train_valid_wd))

    # random permutation train_valid_wd itself for splitting train/valid set
    rng = MersenneTwister()
    Random.shuffle!(rng, train_valid_wd)
    # Befor to prepare (input, output) pairs, determine its size first
    train_valid_wd_size = length(train_valid_wd)
    train_size, valid_size = split_sizes2(train_valid_wd_size, batch_size)

    # split train / valid
    train_wd = train_valid_wd[1:train_size]
    valid_wd = train_valid_wd[(train_size + 1):end]

    # windows dataframe for test (only for 종로구)    
    for name in test_stn_names
        code = seoul_stations[name]
        stn_df = filter_station_DNN(df, code)

        push!(test_wd,
            window_df(stn_df, sample_size, output_size, test_fdate, test_tdate))
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

        target_model, target_statval = train_DNN(train_wd, valid_wd, test_wd,
        target, Symbol(scaled_prefix, target), scaled_train_features, scaling_method,
        train_size, valid_size, test_size,
        sample_size, input_size, batch_size, output_size, epoch_size, eltype,
        statvals, string(target), test_dates)
    end
    #=
    @info "PM10 Training..."
    flush(stdout); flush(stderr)
    
    @info "training feature : " train_features
    @info "sizes (sample, output, epoch, batch) : ", sample_size, output_size, epoch_size, batch_size

    # free minibatch after training because of memory usage
    PM10_model, PM10_statval = train_DNN(train_wd, valid_wd, test_wd,
    :PM10, :scaled_PM10, scaled_train_features, scaling_method,
    train_size, valid_size, test_size,
    sample_size, input_size, batch_size, output_size, epoch_size, eltype,
    statvals, "PM10", test_dates)

    @info "PM25 Training..."
    flush(stdout); flush(stderr)
    
    @info "training feature : " train_features
    @info "sizes (sample, output, epoch, batch) : ", sample_size, output_size, epoch_size, batch_size

    PM25_model, PM25_statval = train_DNN(train_wd, valid_wd, test_wd,
    :PM25, :scaled_PM25, scaled_train_features, scaling_method,
    train_size, valid_size, test_size,
    sample_size, input_size, batch_size, output_size, epoch_size, eltype,
    statvals, "PM25", test_dates)
    =#
end

run_model()
