using Random

using Dates, TimeZones
using MicroLogging

using Mise

function run_model()
    @info "Start Model"
    flush(stdout); flush(stderr)
    df = read_jongro("/input/jongro_single.csv")
    #=
    first(df, 5)
    │ Row │ no     │ stationCode │ date                      │ lat      │ long      │ SO2     │ CO      │ O3      │ NO2     │ PM10    │ PM25    │ temp    │ u       │ v       │ p       │ prep     │ snow     │
    │     │ Int64  │ Int64       │ TimeZones.ZonedDateTime   │ Float64  │ Float64   │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64  │
    ├─────┼────────┼─────────────┼───────────────────────────┼──────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤─────────┤──────────┤──────────┤
    │ 1   │ 1      │ 111123      │ 2015-01-01T01:00:00+09:00 │ 37.572   │ 127.005   │ 0.004   │ 0.2     │ 0.02    │ 0.009   │ 57.0    │ 0.0     │ -7.4    │ 4.41656 │ 1.60749 │ 1011.8  │ missing  │ missing  │
    │ 2   │ 26     │ 111123      │ 2015-01-01T02:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.019   │ 0.008   │ 70.0    │ 3.0     │ -8.0    │ 4.22862 │ 1.53909 │ 1011.7  │ missing  │ missing  │
    │ 3   │ 51     │ 111123      │ 2015-01-01T03:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.02    │ 0.006   │ 92.0    │ 5.0     │ -8.4    │ 3.57083 │ 1.29968 │ 1012.1  │ missing  │ missing  │
    │ 4   │ 76     │ 111123      │ 2015-01-01T04:00:00+09:00 │ 37.572   │ 127.005   │ 0.004   │ 0.2     │ 0.019   │ 0.005   │ 111.0   │ 2.0     │ -8.8    │ 4.60449 │ 1.6759  │ 1012.3  │ missing  │ missing  │
    │ 5   │ 101    │ 111123      │ 2015-01-01T05:00:00+09:00 │ 37.572   │ 127.005   │ 0.005   │ 0.2     │ 0.019   │ 0.006   │ 127.0   │ 5.0     │ -9.1    │ 5.35625 │ 1.94951 │ 1011.8  │ missing  │ missing  │
    =#
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    # For GPU
    default_FloatType::DataType = Float32

    norm_prefix = "norm_"
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]

    μσs = mean_and_std_cols(df, features)
    #hampel!(df, features, norm_features)
    zscore!(df, features, norm_features)

    # convert Float types
    for fea in features
        df[!, fea] = default_FloatType.(df[!, fea])
    end

    for nfea in norm_features
        df[!, nfea] = default_FloatType.(df[!, nfea])
    end

    plot_totaldata(df, :PM25, "/mnt/")
    plot_totaldata(df, :PM10, "/mnt/")

    plot_corr(df, norm_features, features, "/mnt/")
    
    sample_size = 72
    output_size = 24
    epoch_size = 500
    batch_size = 128
    @info "feature : " features
    @info "sizes (sample, output, epoch, batch) : ", sample_size, output_size, epoch_size, batch_size

    # slices for indicies
    # split into segment
    # sg_idxs = split_df(size(df, 1), sample_size)
    # split into window [[1,2,3,4],[2,3,4,5]...]
    train_sdate = ZonedDateTime(2012, 1, 1, 1, tz"Asia/Seoul")
    train_fdate = ZonedDateTime(2017, 12, 31, 23, tz"Asia/Seoul")
    test_sdate = ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul")
    test_fdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

    # windowsed index (index in input)
    train_valid_wd_idxs = window_df(df, sample_size, output_size, train_sdate, train_fdate)
    test_wd_idxs = window_df(df, sample_size, output_size, test_sdate, test_fdate)
    
    # Befor to prepare (input, output) pairs, determine its size first
    train_valid_wd_size = length(train_valid_wd_idxs)
    test_wd_size = length(test_wd_idxs)
    train_size, valid_size = split_sizes2(length(train_valid_wd_idxs), batch_size)
    test_size = length(test_wd_idxs)
    #=
    for idxs in train_valid_wd_idxs
        _dates = df[collect(idxs), :date]
        for _date in _dates
            if _date > test_sdate
                @show "WTF: ", idxs, _date
            end
        end
    end
    =#
    # random permutation train_valid_wd_idxs itself for splitting train/valid set
    rng = MersenneTwister()
    Random.shuffle!(rng, train_valid_wd_idxs)
    # chunks for minibatch, only need train_chnk because of minibatch itself, i.e. 
    train_chnk, valid_chnk = create_chunks(train_valid_wd_idxs, train_size, valid_size, batch_size)
    # pure indexes i.e. [1:2, 2:3, ...]
    train_idxs, valid_idxs = create_idxs(train_valid_wd_idxs, train_size, valid_size)

    # pure indexs
    test_idxs = create_idxs(test_wd_idxs, test_size)
    #test_idxs = collect((train_size + valid_size + 1):(train_size + valid_size + test_size))
    flush(stdout); flush(stderr)
    # simply collect dates, determine exact date for prediction (for 1h, 24h, and so on) later
    test_dates = collect(test_sdate:Hour(1):test_fdate)

    input_size = sample_size * length(features)

    @info "PM10 Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    PM10_model, PM10_μσ = train_DNN(df, :PM10, norm_prefix, norm_features,
    sample_size, input_size, batch_size, output_size, epoch_size, default_FloatType,
    train_valid_wd_idxs, test_wd_idxs, train_chnk, train_idxs, valid_idxs, test_idxs, μσs,
    "PM10", test_dates)

    @info "PM25 Training..."
    flush(stdout); flush(stderr)

    PM25_model, PM25_μσ = train_DNN(df, :PM25, norm_prefix, norm_features,
    sample_size, input_size, batch_size, output_size, epoch_size, default_FloatType,
    train_valid_wd_idxs, test_wd_idxs, train_chnk, train_idxs, valid_idxs, test_idxs, μσs,
    "PM25", test_dates)
    #=
    @info "SO2 Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    SO2_model, SO2_μσ = train_DNN(df, :SO2, norm_prefix, norm_features,
    sample_size, input_size, batch_size, output_size, epoch_size, default_FloatType,
    train_valid_wd_idxs, test_wd_idxs, train_chnk, train_idxs, valid_idxs, test_idxs, μσs,
    "SO2", test_dates)

    @info "CO Training..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    CO_model, CO_μσ = train_DNN(df, :CO, norm_prefix, norm_features,
    sample_size, input_size, batch_size, output_size, epoch_size, default_FloatType,
    train_valid_wd_idxs, test_wd_idxs, train_chnk, train_idxs, valid_idxs, test_idxs, μσs,
    "CO", test_dates)

    @info "NO2 Training..."
    flush(stdout); flush(stderr)

    NO2_model, NO2_μσ = train_DNN(df, :NO2, norm_prefix, norm_features,
    sample_size, input_size, batch_size, output_size, epoch_size, default_FloatType,
    train_valid_wd_idxs, test_wd_idxs, train_chnk, train_idxs, valid_idxs, test_idxs, μσs,
    "NO2", test_dates)
    =#
end

run_model()
