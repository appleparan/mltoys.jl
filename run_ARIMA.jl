using Random

using Dates, TimeZones
using MicroLogging
using DataFrames, Query, CSV

using StatsBase, Statistics
using TimeSeries
using HypothesisTests
using Loess
using StateSpaceModels
using Mise

function run_arima()

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
    train_fdate = ZonedDateTime(2008, 1, 1, 1, tz"Asia/Seoul")
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
    #train_stn_names = [:종로구, :강서구, :송파구, :강남구]
    
    train_stn_names = [
        :중구,:종로구,:용산구,:광진구,:성동구,
        :중랑구,:동대문구,:성북구,:도봉구,:은평구,
        :서대문구,:마포구,:강서구,:구로구,:영등포구,
        :동작구,:관악구,:강남구,:서초구,:송파구,
        :강동구,:금천구,:강북구,:강북구,:양천구,
        :양천구,:노원구]
    train_stn_names = [:종로구]

    df = filter_raw_data(df, train_fdate, train_tdate, test_fdate, test_tdate)
    @show first(df, 10)

    features = [:PM10, :PM25]
    # If you want exclude some features, modify train_features
    # exclude :PM10, :PM25 temporarily for log transform
    train_features = [:PM10, :PM25]
    target_features = [:PM10, :PM25]

    # For GPU, change precision of Floating numbers
    eltype::DataType = Float32
   
    # simply collect dates, determine exact date for prediction (for 1h, 24h, and so on) later

    Base.Filesystem.mkpath("/mnt/ARIMA/")

    for ycol in target_features
        for name in train_stn_names
            Base.Filesystem.mkpath("/mnt/ARIMA/$(string(ycol))")
            code = seoul_stations[name]
            stn_df = filter_raw_data(df, code, train_fdate, train_tdate)
            dates = stn_df[!, :date]

            # Imputation
            # allow missing again
            DataFrames.allowmissing!(stn_df)
            zero2Missing!(stn_df, ycol)
            impute!(stn_df, ycol, :sample)
            # disallow missing again
            DataFrames.disallowmissing!(stn_df)
            ycols = stn_df[!, ycol]
            logycols = log10.(stn_df[!, ycol])
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_impute") => ycols)
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_logimpute") => logycols)

            ta = TimeArray(dates, stn_df[!, ycol])

            total_mean, total_std = StatsBase.mean_and_std(values(ta))

            @info "Total mean of $(string(ycol)) : ", total_mean
            @info "Total std  of $(string(ycol)) : ", total_std

            # multiple seasonality analysis

            # Create structural model with seasonality of 365 x 24
            
            #model = structural(logycols, 365*24)
            #ss = statespace(model)
            
        end
    end
end

run_arima()