using Random

using Dates, TimeZones
using MicroLogging
using DataFramesMeta

using Mise

ENV["GKSwstype"] = "nul"
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
    default_FloatType::DataType = Float64

    norm_prefix = "norm_"
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]

    μσs = mean_and_std_cols(df, features)
    zscore!(df, features, norm_features)

    # convert Float types
    for fea in features
        df[!, fea] = default_FloatType.(df[!, fea])
    end

    for nfea in norm_features
        df[!, nfea] = default_FloatType.(df[!, nfea])
    end

    train_features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]

    norm_train_features = [Symbol(eval(norm_prefix * String(f))) for f in train_features]

    plot_totaldata(df, :PM25, "/mnt/")
    plot_totaldata(df, :PM10, "/mnt/")

    plot_pcorr(df, norm_features, features, "/mnt/")

    sample_size = 72
    output_size = 24
    hid_size = 64
    batch_size = 32
    epoch_size = 500
    
    @info "training feature : " train_features
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

    df_test = DataFramesMeta.@linq df |>
                    where(:date .>= test_sdate, :date .<= test_fdate)

    @info "PM10 Evolving..."
    flush(stdout); flush(stderr)

    # free minibatch after training because of memory usage
    evolve_OU(df_test, :PM10, μσs, default_FloatType, test_dates)

    @info "PM25 Evolving..."
    flush(stdout); flush(stderr)

    evolve_OU(df_test, :PM25, μσs, default_FloatType, test_dates)
end

run_model()
