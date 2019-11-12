using Random

using Dates, TimeZones
using MicroLogging

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
    total_fdate, total_tdate = get_date_range(df)
    train_fdate = ZonedDateTime(2012, 1, 1, 1, tz"Asia/Seoul")
    train_tdate = ZonedDateTime(2017, 12, 31, 23, tz"Asia/Seoul")
    test_fdate = ZonedDateTime(2018, 1, 1, 0, tz"Asia/Seoul")
    test_tdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

    # stations (LSTNet lmit 1 station)
    train_stn_names = [:종로구, :광진구, :강서구, :강남구]
    train_stn_names = [:종로구, :강서구]
    #train_stn_names = [:종로구]

    # test set is Jongro only
    test_stn_names = [:종로구]
    
    df = filter_raw_data(df, train_fdate, train_tdate, test_fdate, test_tdate)
    @show first(df, 10)

    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]

    # For GPU
    eltype::DataType = Float32

    norm_prefix = "norm_"
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]

    # I want exclude some features, modify train_features
    train_features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_train_features = [Symbol(eval(norm_prefix * String(f))) for f in train_features]

    μσs = mean_and_std_cols(df, train_features)
    zscore!(df, features, norm_features)
    #min_max_scaling!(df, train_features, norm_train_features, 0.0, 10.0)

    # convert Float types
    for fea in features
        df[!, fea] = eltype.(df[!, fea])
    end

    for nfea in norm_features
        df[!, nfea] = eltype.(df[!, nfea])
    end

    sample_size = 48
    output_size = 24
    # kernel_length  hours for extract locality 
    kernel_length = 3
    epoch_size = 300
    batch_size = 32

    # windowed dataframe for train/valid (input + output)
    train_valid_wd = []
    test_wd = []

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

    @info "PM10 Training..."
    flush(stdout); flush(stderr)
    
    @info "training feature : " train_features
    @info "sizes (sample, output, kernel, epoch, batch) : ",
        sample_size, output_size, kernel_length, epoch_size, batch_size

    # free minibatch after training because of memory usage
    PM10_model, PM10_μσ = train_LSTNet(train_wd, valid_wd, test_wd,
    :PM10, norm_prefix, train_features,
    train_size, valid_size, test_size,
    sample_size, batch_size, kernel_length, output_size, epoch_size, eltype,
    μσs, "PM10", test_dates)

    @info "PM25 Training..."
    flush(stdout); flush(stderr)
    
    @info "training feature : " train_features
    @info "sizes (sample, output, epoch, batch) : ", sample_size, output_size, epoch_size, batch_size

    PM25_model, PM25_μσ = train_LSTNet(train_wd, valid_wd, test_wd,
    :PM25, norm_prefix, train_features,
    train_size, valid_size, test_size,
    sample_size, batch_size, kernel_length, output_size, epoch_size, eltype,
    μσs, "PM25", test_dates)
end

run_model()
