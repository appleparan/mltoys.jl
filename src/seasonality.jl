"""
    periodic_mean(df, target, pdir)

Filter DataFrame df by time directive and get averages

time directive must be in :hour (24 hour), :day (365 day), :quarter (4 quarter)
"""
function periodic_mean(df::DataFrame, target::Symbol, pdir::Symbol)
    if pdir == :hour
        max_period = 24
    elseif pdir == :week
        max_period = 53
    elseif pdir == :month
        max_period = 12
    else
        error("Period directive must be between :hour, :day, :month, :quarter")
    end

    avgs = zeros(max_period)

    for period=1:max_period
        if pdir == :hour
            _df = @from i in df begin
                @where hour(i.date) == period - 1
                @orderby i.date
                @select i
                @collect DataFrame
            end
        elseif pdir == :week
            _df = @from i in df begin
                @where week(i.date) == period
                @orderby i.date
                @select i
                @collect DataFrame
            end
        elseif pdir == :month
            _df = @from i in df begin
                @where month(i.date) == period
                @orderby i.date
                @select i
                @collect DataFrame
            end
        end

        avgs[period] = mean(_df[!, target])
    end

    avgs
end

"""
    periodic_mean(df, target, pdir)

decompose double seasonality

time directive must be in :hour (24 hour), :day (365 day), :quarter (4 quarter)
"""
function populate_periodic_mean(df::DataFrame, target::Symbol, pdir::Symbol)
    means = periodic_mean(df, target, pdir)

    # adjust index only for hourly (0-23)
    if pdir == :hour
        means_repeated = means[df[!, pdir] .+ 1]
    else
        means_repeated = means[df[!, pdir]]
    end

    means, means_repeated
end

function smoothing_series(xs, ys; span = 0.75)
    model = Loess.loess(xs, ys; span = span)

    us = collect(range(extrema(xs)...; step = 1))
    vs = Loess.predict(model, us)

    us, vs
end

function compute_annual_mean(df::DataFrame, ycol::Symbol)
    # to deal with leap year, just make large array for indexing
    # when moving window have some error but it is ignorable
    year_wd_mean = zeros(12*31*24)
    year_wd_count = zeros(Int64, 12*31*24)

    last_date = df[end, :date]
    window = 24

    # mean of window
    mean_vals = map(i -> mean(@view df[i:i+(window-1), ycol]), 1:size(df, 1)-window+1)
    for (i, row) in enumerate(eachrow(df))
        t = getproperty(row, :date)
        if t > last_date - Dates.Hour(window)
            break
        end
        v = mean_vals[i]

        idx = (Dates.month(t) - 1) * 31 * 24 + (Dates.day(t) - 1) * 24 + Dates.hour(t) + 1
        year_wd_count[idx] += 1
        year_wd_mean[idx] = year_wd_mean[idx] + (v - year_wd_mean[idx]) / year_wd_count[idx]
    end

    year_wd_mean
end

"""
    season_adj_lee(df::DataFrame, ycol::Symbol)
simple implementation for Prof. Lee's seasonal adjustment

Assume data as

y = smoothing(S_year) + S_day + residual

S_year : annual seasonality
S_day : daily seasonality
residual : residual

`S_year` and `S_day` is computed by mean
naive but easy to implement
"""
function season_adj_lee(df::DataFrame, ycol::Symbol)
    f_date = df[1, :date]
    t_date = df[end, :date]

    season_adj_lee(df, ycol, f_date, t_date)
end

"""
    season_adj_lee(df::DataFrame, ycol::Symbol, f_date::DateTime, t_date::DateTime)
Prof. Lee's seasonal adjustment with date range
station fluctuation will be averaged

Assume data as

y = smoothing(S_year) + S_day + residual

S_year : annual seasonality
S_day : daily seasonality
residual : residual

`S_year` and `S_day` is computed by mean
naive but easy to implement
"""
season_adj_lee(df::DataFrame, ycol::Symbol,
    f_date::ZonedDateTime, t_date::ZonedDateTime) =
    season_adj_lee(df, ycol, DateTime(f_date, Local), DateTime(t_date, Local))

function season_adj_lee(df::DataFrame, ycol::Symbol,
    f_date::DateTime, t_date::DateTime)

    # filter by given date range
    _df = @from i in df begin
        @where f_date <= DateTime(i.date, Local) <= t_date
        @orderby i.date
        @select i
        @collect DataFrame
    end

    # all station values are averaged by date and filter date, ycol, and mean_ycol only
    ycol_mean = Symbol(ycol, "_mean")

    # Metaprogamming for ycol_maen = ycol => x -> mean
    # https://discourse.julialang.org/t/metaprogramming-function-with-keyword-argument/12551/2?u=appleparan
    z = Expr(:kw, ycol_mean, ycol => mean)
    period_df = eval(:(DataFrames.by($(_df), :date; $z)))
    # insert original columns

    last_date = period_df[end, :date]
    window = 24

    # window mean
    year_wd_mean = compute_annual_mean(period_df, ycol_mean)
    # remove annual seasonality
    # fluctuation of annual average by 24 hour window
    year_res1 = zeros(length(period_df[!, :date]))
    year_sea1 = zeros(length(period_df[!, :date]))

    for (i, row) in enumerate(eachrow(period_df))
        t = getproperty(row, :date)
        if t > last_date - Dates.Hour(window)
            break
        end
        idx = (Dates.month(t) - 1) * 31 * 24 + (Dates.day(t) - 1) * 24 + Dates.hour(t) + 1

        year_res1[i] = getproperty(row, ycol_mean) - year_wd_mean[idx]
        year_sea1[i] = year_wd_mean[idx]
    end

    # smoothing(S_year)
    # The leap year error will be ignorable thanks to smoothing
    year_sea2_x, year_sea2_y = smoothing_series(float.(collect(1:length(year_sea1))), year_sea1; span = 0.1)

    #_model = Loess.loess(float.(collect(1:length(year_sea1))), year_sea1)
    #year_sea2_x = collect(range(extrema(float.(collect(1:length(year_sea1))))...; step = 1))
    #year_sea2_y = Loess.predict(_model, year_sea2_x)

    # Interpolations
    #itp = interpolate((year_sea2_x, year_sea2_y), BSpline(Cubic(Line(OnGrid()))))
    #year_sea2 = map(i -> itp(float(i)), 1:length(period_df[!, :date]))
    # Dierckx
    spl = Spline1D(year_sea2_x, year_sea2_y)
    year_sea2 = evaluate(spl, float.(collect(1:length(period_df[!, :date]))))
    year_res2 = period_df[!, ycol_mean] .- year_sea2
    DataFrames.insertcols!(period_df, 3, :year_sea2 => year_sea2)
    DataFrames.insertcols!(period_df, 3, :year_res2 => year_res2)

    # now calculate S_day
    hour_col = hour.(period_df[!, :date])
    DataFrames.insertcols!(period_df, 3, :hour => hour_col)

    # put mean values to dataframe, adjust index only for hour (0-23)
    day_sea1_summary, day_sea1 = populate_periodic_mean(period_df, :year_res2, :hour)
    DataFrames.insertcols!(period_df, 3, :day_sea1 => day_sea1)
    day_res1 = zeros(length(period_df[!, :date]))

    for (i, row) in enumerate(eachrow(period_df))
        t = getproperty(row, :date)
        if t > last_date - Dates.Hour(window)
            break
        end

        day_res1[i] = getproperty(row, :year_res2) - getproperty(row, :day_sea1)
    end

    # S_year, smoothed(S_year), S_day, R
    year_sea1, year_sea2, year_res2, day_sea1, day_res1
end
#=
"""
    bsts()

Decompose series by Bayesian structural time series models

```math
    yₜ = μₜ + τₜ + βₜ x + ϵₜ

    where ϵₜ ~ N(0, σ²_ϵ  )
```
μₜ : local linear trend
τₜ : seasonality
βₜ x : regression component
ϵₜ : noise

# Arguments
- level::Symbol, optional : Whether or not to include a level component.
- trend::Bool, optional : Whether or not to include a trend component. If true, level must not be irregular
- seaasonal::Integer, optional : The period of the seasonal component
- freq_seasonal::Array{NamedTuple,1} : Whether (and how) to model seasonal component(s) with trig
    form of Array of NamedTuple, like (period = some_value, harmonics = some_value)
    period must be present
- cycle::Bool : Whether or not to include a cycle component
- autoregressive::Integer : The order of the autoregressive component
- exog::Array{Float64,1} : Exogenous variables
- irregular::Bool : Whether or not to include an irregular component
- stochastic_level : Whether or not any level component is stochastic
- stochastic_trend : Whether or not any trend component is stochastic
- stochastic_seasonal : Whether or not any seasonal component is stochastic
- stochastic_freq_seasonal : Whether or not each seasonal component(s) is (are) stochastic
- stochastic_cycle : Whether or not any cycle component is stochastic.
- damped_cycle : Whether or not the cycle component is damped

https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
"""
function bsts(;level::Symbol=:irregular, trend::Bool=false, seasonal::Integer=0,
    freq_seasonal::Array{NamedTuple,1}=[(period=0., harmonics=0.)],
    cycle::Bool=false, autoregressive::Integer=0,
    exog::Array{Float64,1}=[], irregular::Bool=false,
    stochastic_level::Bool=false, stochastic_seasonal::Bool=false,
    stochastic_freq_seasonal::Array{Bool, 1}, stochastic_cycle::Bool=false,
    damped_cycle::Bool=false, cycle_period_bounds::Tuple{Float64, Float64}=(0.5, Inf),
    use_exact_diffuse::Bool=false)

    # Local linear trend

end
=#