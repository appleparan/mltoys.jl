using Random

using Dates, TimeZones
using MicroLogging
using NumericalIntegration
using DataFrames, DataFramesMeta

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
            dates = stn_df[!, :date]

            ta = TimeArray(dates, stn_df[!, ycol])

            total_mean, total_std = StatsBase.mean_and_std(values(ta))

            @info "Total mean of $(string(ycol)) : ", total_mean
            @info "Total std  of $(string(ycol)) : ", total_std
            
            # Analysis : pdf
            @info "Estimate probability density function..."
            ycol_arr = stn_df[!, ycol] .+ 10
            ycol_pdf = pdf(ycol_arr)
            plot_anal_pdf(ycol_pdf, ycol,
                "PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_pdf_zeromissing")

            # disabled due to nonpositive bandwidth of pdf estimation
            ycol_logpdf = pdf(log10.(ycol_arr))
            plot_anal_pdf(ycol_logpdf, ycol,
                "Log PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_logpdf_zeromissing")
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_zeromissing") => ycol_arr)
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_log_zeromissing") => log10.(ycol_arr))

            # missing value to total_mean 
            ycol_arr = stn_df[!, ycol]
            replace!(ycol_arr, 0.0 => Int(round(total_mean)))

            ycol_pdf = pdf(ycol_arr)
            plot_anal_pdf(ycol_pdf, ycol,
                "PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_pdf_meanmissing")
            ycol_logpdf = pdf(log10.(ycol_arr))
            plot_anal_pdf(ycol_logpdf, ycol,
                "Log PDF for $(ycol)", "/mnt/analysis/", "$(string(ycol))_$(string(name))_logpdf_meanmissing")
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_meanmissing") => ycol_arr)
            DataFrames.insertcols!(stn_df, 3, Symbol(ycol, "_log_meanmissing") => log10.(ycol_arr))

            # Analysis : ADFTest (Check Stationary)
            max_lag = 30 * 24
            @show HypothesisTests.ADFTest(values(ta), :none, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :constant, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :trend, max_lag)
            @show HypothesisTests.ADFTest(values(ta), :squared_trend, max_lag)

            # Analysis : Autocorrelation Functions
            max_time = Int(ceil(min(size(stn_df[!, ycol],1)-1, 10*log10(size(stn_df[!, ycol],1)))))
            @info "Autocorrelation..."
            lags = 0:max_time
            acf = StatsBase.autocor(values(ta), lags)
            plot_anal_correlogram(acf, ycol, 
                "Autocorrelation",
                "/mnt/analysis/", "$(string(ycol))_$(string(name))_mt$(max_time)_acf")

            pacf = StatsBase.pacf(values(ta), lags)
            plot_anal_correlogram(pacf, ycol,
                "Partial Autocorrelation",
                "/mnt/analysis/", "$(string(ycol))_$(string(name))_mt$(max_time)_pacf")
            #=
            # Analysis : Autocorrelation Functions
            # No need to compute pacf
            max_time = 400 * 24
            lags = 0:max_time
            acf = StatsBase.autocor(values(ta), lags)
            plot_anal_correlogram(acf, ycol, 
                "Autocorrelation",
                "/mnt/analysis/", "$(string(ycol))_$(string(name))_mt$(max_time)_acf")

            # Analysis : Integral Length Scale (Total)
            int_scale = integrate(1:length(acf), acf, SimpsonEven())
            @info "Integral Time Scale of $(ycol) at $(name) (total): ", int_scale

            plot_anal_autocor(acf, ycol,
                "Autocorrelation (total series)",
                "/mnt/analysis/", "$(string(ycol))_$(string(name))_total")
            =#
            # Analysis : Time difference
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

            # Analysis : Grouping by time
            @info "Groupby time..."

            for _ycol in [ycol, Symbol(ycol, "_zeromissing"), Symbol(ycol, "_log_zeromissing"), Symbol(ycol, "_meanmissing"), Symbol(ycol, "_log_meanmissing")]
                time_grp_df = copy(stn_df)
                # hourly
                hourlycol = hour.(time_grp_df[!, :date])
                DataFrames.insertcols!(time_grp_df, 3, :hour => hourlycol)
                hourly_grp_means = time_mean(time_grp_df, _ycol, :hour)

                # put mean values to dataframe, adjust index only for hour (0-23)
                hourly_grp_means_repeated = hourly_grp_means[time_grp_df[!, :hour] .+ 1]
                DataFrames.insertcols!(time_grp_df, 3, :hour_mean => hourly_grp_means_repeated)
                
                plot_anal_violin(time_grp_df, _ycol, :hour, hourly_grp_means,
                    "Violin plot for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))")
                plot_anal_time_mean(time_grp_df, _ycol, :hour, hourly_grp_means,
                    "Hourly mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_mean")

                # daily
                dailycol = dayofyear.(time_grp_df[!, :date])
                DataFrames.insertcols!(time_grp_df, 3, :day => dailycol)
                daily_grp_means = time_mean(time_grp_df, _ycol, :day)

                # put mean values to dataframe
                daily_grp_means_repeated = daily_grp_means[time_grp_df[!, :day]]
                DataFrames.insertcols!(time_grp_df, 3, :day_mean => daily_grp_means_repeated)

                plot_anal_violin(time_grp_df, _ycol, :day, daily_grp_means,
                    "Violin plot for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))")
                plot_anal_time_mean(time_grp_df, _ycol, :day, daily_grp_means,
                    "Daily mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_mean")
            
                # monthly
                monthlycol = month.(time_grp_df[!, :date])
                DataFrames.insertcols!(time_grp_df, 3, :month => monthlycol)
                monthly_grp_means = time_mean(time_grp_df, _ycol, :month)

                # put mean values to dataframe
                monthly_grp_means_repeated = monthly_grp_means[time_grp_df[!, :month]]
                DataFrames.insertcols!(time_grp_df, 3, :month_mean => monthly_grp_means_repeated)

                plot_anal_violin(time_grp_df, _ycol, :month, monthly_grp_means,
                    "Violin plot for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))")
                plot_anal_time_mean(time_grp_df, _ycol, :month, monthly_grp_means,
                    "Monthly mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_mean")

                # quarterly
                quarterlycol = quarterofyear.(time_grp_df[!, :date])
                DataFrames.insertcols!(time_grp_df, 3, :quarter => quarterlycol)
                quarterly_grp_means = time_mean(time_grp_df, _ycol, :quarter)
                quarterly_grp_means_repeated = quarterly_grp_means[time_grp_df[!, :quarter]]
                DataFrames.insertcols!(time_grp_df, 3, :quarter_mean => quarterly_grp_means_repeated)

                plot_anal_violin(time_grp_df, _ycol, :quarter, quarterly_grp_means,
                    "Violin plot for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))")
                plot_anal_time_mean(time_grp_df, _ycol, :quarter, quarterly_grp_means,
                    "Quarterly mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_mean")

                # Analysis : fluctuation analysis, hourly, monthly, quarterly
                # We don't need daily 
                # Use Split-Apply-Combine Strategy
                @info "Time fluctuations..."
                hourlyfluc_df = DataFrames.by(time_grp_df, _ycol) do _df
                    (hour = _df.hour,
                    date = _df.date,
                    fluc = getproperty(_df, _ycol) .- getproperty(_df, Symbol(:hour, "_mean")) )
                end
                hourlyfluc_grp_means = time_mean(hourlyfluc_df, _ycol, :hour)
                #plot_anal_time_fluc(hourlyfluc_df, :hour, _ycol,
                #    "Hourly fluctuation for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_fluc")
                plot_anal_time_flucmean(hourlyfluc_df, _ycol, :hour, hourlyfluc_grp_means, 
                    "Daily fluctuation mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_flucmean")
                
                dailyfluc_df = DataFrames.by(time_grp_df, _ycol) do _df
                    (day = _df.day,
                    date = _df.date,
                    fluc = getproperty(_df, _ycol) .- getproperty(_df, Symbol(:day, "_mean")) )
                end
                dailyfluc_grp_means = time_mean(dailyfluc_df, _ycol, :day)
                #plot_anal_time_fluc(dailyfluc_df, :day, _ycol,
                #    "Daily fluctuation for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_fluc")
                plot_anal_time_flucmean(dailyfluc_df, _ycol, :day, dailyfluc_grp_means, 
                    "Daily fluctuation mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_flucmean")

                monthlyfluc_df = DataFrames.by(time_grp_df, _ycol) do _df
                    (month = _df.month,
                    date = _df.date,
                    fluc = getproperty(_df, _ycol) .- getproperty(_df, Symbol(:month, "_mean")) )
                end
                monthlyfluc_grp_means = time_mean(monthlyfluc_df, _ycol, :month)
                #plot_anal_time_fluc(monthlyfluc_df, :month, _ycol,
                #    "Monthly fluctuation for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_fluc")
                plot_anal_time_flucmean(monthlyfluc_df, _ycol, :month, monthlyfluc_grp_means, 
                    "Monthly fluctuation mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_flucmean")

                quarterlyfluc_df = DataFrames.by(time_grp_df, _ycol) do _df
                    (quarter = _df.quarter,
                    date = _df.date,
                    fluc = getproperty(_df, _ycol) .- getproperty(_df, Symbol(:quarter, "_mean")) )
                end
                quarterlyfluc_grp_means = time_mean(quarterlyfluc_df, _ycol, :quarter)
                #plot_anal_time_fluc(quarterlyfluc_df, :quarter, _ycol,
                #    "Quarterly fluctuation for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_fluc")
                plot_anal_time_flucmean(quarterlyfluc_df, _ycol, :quarter, quarterlyfluc_grp_means, 
                    "Quarterly fluctuation mean for $(ycol)", "/mnt/analysis/", "$(string(_ycol))_$(string(name))_time_flucmean")
            end
        end
    end

end

run_analysis()
