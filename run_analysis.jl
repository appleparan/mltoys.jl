using Random

using Dates, TimeZones
using MicroLogging
using NumericalIntegration
using DataFrames, DataFramesMeta, CSV

using StatsBase
using TimeSeries
using HypothesisTests
using Mise
using StatsPlots

function run_analysis()

    @info "Start Analysis"
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

    Base.Filesystem.mkpath("/mnt/analysis/")

    for ycol in target_features
        # DNN's windowed dataframe is already filterred with dates.
        # size(df) = (sample_size, length(features))
        @info "Analysis of $(string(ycol))"
        for name in train_stn_names
            code = seoul_stations[name]
            stn_df = filter_raw_data(df, code, train_fdate, train_tdate)
            # for imputatation
            CSV.write("/mnt/analysis/raw_data_analysis_$(string(ycol))_$((name)).csv", stn_df)
            dates = stn_df[!, :date]

            # Imputation
            # allow missing again
            DataFrames.allowmissing!(stn_df)

            zero2Missing!(stn_df, ycol)

            # Analysis : pdf
            @info "Estimate probability density function without imputation..."
            ycol_arr = collect(skipmissing(stn_df[!, ycol]))
            @show ycol_arr[1:100]
            if ycol == :PM10
                npts = 64
            elseif ycol == :PM25
                npts = 16
            end
            ycol_pdf = pdf(ycol_arr, npts)
            plot_anal_pdf(ycol_pdf, ycol,
                "PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_pdf_noimpute")
            ycol_logpdf = pdf(log10.(ycol_arr), npts)
            plot_anal_pdf(ycol_logpdf, ycol,
                "Log PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_logpdf_noimpute")

            # mean imputation
            #impute!(df, ycol, :mean; total_mean = total_mean)
            # simple sampling
            impute!(stn_df, ycol, :sample)
            # knn imputation
            #nK = 10
            #ycols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
            #impute!(stn_df, ycol, :knn; ycols = ycols, leafsize = nK)

            # disallow missing again
            DataFrames.disallowmissing!(stn_df)

            ta = TimeArray(dates, stn_df[!, ycol])

            @info "Estimate probability density function without imputation..."
            ycol_arr = stn_df[!, ycol]

            total_mean, total_std = StatsBase.mean_and_std(values(ta))

            @info "Total mean of $(string(ycol)) : ", total_mean
            @info "Total std  of $(string(ycol)) : ", total_std

            if ycol == :PM10
                npts = 64
            elseif ycol == :PM25
                npts = 16
            end
            ycol_pdf = pdf(ycol_arr, npts)
            plot_anal_pdf(ycol_pdf, ycol,
                "PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_pdf_impute")
            ycol_logpdf = pdf(log10.(ycol_arr), npts)
            plot_anal_pdf(ycol_logpdf, ycol,
                "Log PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_logpdf_impute")
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_impute") => ycol_arr)
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_logimpute") => log10.(ycol_arr))

            # Analysis : ADFTest (Check Stationary)
            max_lag = 30 * 24
            @show HypothesisTests.ADFTest(values(ta), :none, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :constant, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :trend, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :squared_trend, max_lag)

            # Analysis : Autocorrelation Functions]
            for _ycol in [ycol, Symbol(ycol, :_impute), Symbol(ycol, :_logimpute)]
                Base.Filesystem.mkpath("/mnt/analysis/$(string(_ycol))/")
                max_time = Int(ceil(min(size(stn_df[!, ycol],1)-1, 10*log10(size(stn_df[!, ycol],1)))))
                @info "Autocorrelation..."
                lags = 0:max_time
                acf = StatsBase.autocor(stn_df[!, _ycol], lags)
                plot_anal_correlogram(acf, _ycol, 
                    "Autocorrelation",
                    "/mnt/analysis/$(string(_ycol))/", "$(string(ycol))_$(string(name))_mt$(max_time)_acf")

                pacf = StatsBase.pacf(stn_df[!, _ycol], lags)
                plot_anal_correlogram(pacf, _ycol,
                    "Partial Autocorrelation",
                    "/mnt/analysis/$(string(_ycol))/", "$(string(ycol))_$(string(name))_mt$(max_time)_pacf")

                # Analysis : Autocorrelation Functions
                # No need to compute pacf
                max_time = 400 * 24
                lags = 0:max_time
                acf = StatsBase.autocor(stn_df[!, _ycol], lags)
                plot_anal_correlogram(acf, ycol, 
                    "Autocorrelation",
                    "/mnt/analysis/$(string(_ycol))/", "$(string(ycol))_$(string(name))_mt$(max_time)_acf")
            end
            #=
            # Analysis : Integral Length Scale (Total)
            int_scale = integrate(1:length(acf), acf, SimpsonEven())
            @info "Integral Time Scale of $(ycol) at $(name) (total): ", int_scale

            plot_anal_autocor(acf, ycol,
                "Autocorrelation (total series)",
                "/mnt/analysis/", "$(string(ycol))_$(string(name))_total")
            =#
            # Analysis : Time difference
            #=
            @info "Time difference..."
            order = 1
            lag = 1
            diffed = TimeSeries.diff(ta, lag, padding=true, differences=order)
            val_diffed = copy(values(diffed))
            replace!(val_diffed, Inf => 0, -Inf => 0, NaN => 0)
            @info "mean of time $(order)th difference of $(ycol) at $(name): ", StatsBase.mean(val_diffed)
            @info "std  of time $(order)th difference of $(ycol) at $(name): ", StatsBase.std(val_diffed)
            plot_anal_lineplot(DateTime.(timestamp(diffed)), val_diffed, ycol,
                order, lag, "/mnt/analysis/", "$(string(ycol))_$(string(name))")

            order = 2
            lag = 1
            diffed = TimeSeries.diff(ta, lag, padding=true, differences=order)
            val_diffed = copy(values(diffed))
            replace!(val_diffed, Inf => 0, -Inf => 0, NaN => 0)
            @info "mean of time $(order)th difference of $(ycol) at $(name): ", StatsBase.mean(val_diffed)
            @info "std  of time $(order)th difference of $(ycol) at $(name): ", StatsBase.std(val_diffed)
            plot_anal_lineplot(DateTime.(timestamp(diffed)), val_diffed, ycol,
                order, lag, "/mnt/analysis/", "$(string(ycol))_$(string(name))")

            order = 1
            lag = 24
            diffed = TimeSeries.diff(ta, lag, padding=true, differences=order)
            val_diffed = copy(values(diffed))
            replace!(val_diffed, Inf => 0, -Inf => 0, NaN => 0)
            @info "mean of time $(order)th difference of $(ycol) at $(name): ", StatsBase.mean(val_diffed)
            @info "std  of time $(order)th difference of $(ycol) at $(name): ", StatsBase.std(val_diffed)
            plot_anal_lineplot(DateTime.(timestamp(diffed)), val_diffed, ycol,
                order, lag, "/mnt/analysis/", "$(string(ycol))_$(string(name))")
            =#
            # Analysis : Grouping by time
            @info "Groupby time..."

            for _ycol in [ycol, Symbol(ycol, :_impute), Symbol(ycol, :_logimpute)]
                time_grp_df = copy(stn_df)
                Base.Filesystem.mkpath("/mnt/analysis/$(string(_ycol))/")
                # find periodicty
                tfuncs = (
                    hour = :hour,
                    week = :week,
                    month = :month,
                    quarter = :quarterofyear)

                # Analysis : mean analysis, hourly, weekly, monthly, quarterly
                for time_dir in [:hour, :week, :month, :quarter] 
                    @info "Time mean of $(string(_ycol)) $(string(time_dir))ly... "
                    tfunc = tfuncs[time_dir]
                    timely_col = eval(tfunc).(time_grp_df[!, :date])
                    DataFrames.insertcols!(time_grp_df, 3, time_dir => timely_col)
                    timely_grp_means = time_mean(time_grp_df, _ycol, time_dir)

                    # put mean values to dataframe, adjust index only for hour (0-23)
                    if time_dir == :hour
                        timely_grp_means_repeated = timely_grp_means[time_grp_df[!, time_dir] .+ 1]
                    else 
                        timely_grp_means_repeated = timely_grp_means[time_grp_df[!, time_dir]]
                    end
                    DataFrames.insertcols!(time_grp_df, 3, Symbol(time_dir, "_mean") => timely_grp_means_repeated)
                    # fluctuations
                    DataFrames.insertcols!(time_grp_df, 3, Symbol(time_dir, "_fluc") => time_grp_df[!, ycol] .- time_grp_df[!, Symbol(time_dir, "_mean") ])
                    
                    plot_anal_violin(time_grp_df, _ycol, time_dir, timely_grp_means,
                        "Violin plot for $(ycol)", "/mnt/analysis/$(string(_ycol))/", "$(string(ycol))_$(string(name))")
                    plot_anal_time_mean(time_grp_df, _ycol, time_dir, timely_grp_means,
                        "$(string(time_dir))ly mean for $(ycol)", "/mnt/analysis/$(string(_ycol))/", "$(string(ycol))_$(string(name))_time_mean")
                end

                # Analysis : fluctuation analysis, hourly, weekly, monthly, quarterly
                # Use Split-Apply-Combine Strategy
                
                for time_dir in [:hour, :week, :month, :quarter] 
                    @info "Time fluctuations of $(string(_ycol)) $(string(time_dir))ly..."

                    timelyfluc_df = DataFrames.by(time_grp_df, _ycol) do _df
                        (; time_dir => getproperty(_df, time_dir),
                        :date => _df.date,
                        :mean => getproperty(_df, Symbol(time_dir, "_mean")),
                        :fluc => getproperty(_df, _ycol) - getproperty(_df, Symbol(time_dir, "_mean")))
                    end

                    timelyfluc_grp_means = time_mean(timelyfluc_df, _ycol, time_dir)
                    plot_anal_time_mean(timelyfluc_df, _ycol, time_dir, timelyfluc_grp_means, 
                        "$(uppercasefirst(string(time_dir)))ly fluctuation mean for $(ycol)", "/mnt/analysis/$(string(_ycol))/",
                        "$(string(ycol))_$(string(name))_$(string(_ycol))_time_flucmean")

                    max_time = Int(ceil(min(size(timelyfluc_df[!, _ycol],1)-1, 10*log10(size(timelyfluc_df[!, _ycol],1)))))
                    lags = 0:max_time
                    @info "$(string(time_dir))ly Fluctuation Autocorrelation..."
                    acf = StatsBase.autocor(timelyfluc_df[!, :fluc], lags)
                    plot_anal_correlogram(acf, _ycol, 
                        "$(uppercasefirst(string(time_dir)))ly Fluctuation Autocorrelation",
                        "/mnt/analysis/$(string(_ycol))/",
                        "$(string(ycol))_$(string(name))_mt$(max_time)_fluc_$(string(time_dir))ly_acf")

                    max_time = 400 * 24
                    lags = 0:max_time
                    acf = StatsBase.autocor(timelyfluc_df[!, :fluc], lags)
                    plot_anal_correlogram(acf, _ycol, 
                        "$(uppercasefirst(string(time_dir)))ly Fluctuation Autocorrelation",
                        "/mnt/analysis/$(string(_ycol))/",
                        "$(string(ycol))_$(string(name))_mt$(max_time)_fluc_$(string(time_dir))ly_acf")
                end
            end
        end
    end

end

run_analysis()
