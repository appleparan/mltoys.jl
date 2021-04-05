# Opencolor : https://yeun.github.io/open-color/
const GRAY4 = colorant"#CED4dA"
const GRAY7 = colorant"#495057"
const RED4 = colorant"#FF8787"
const RED7 = colorant"#F03E3E"
const PINK4 = colorant"#F783AC"
const GRAPE4 = colorant"#DA77F2"
const VIOLET4 = colorant"#9775FA"
const INDIGO4 = colorant"#748FFC"
const BLUE4 = colorant"#4DABF7"
const CYAN4 = colorant"#3BC9DB"
const TEAL4 = colorant"#38D9A9"
const GREEN4 = colorant"#69DB7C"
const LIME4 = colorant"#A9E34b"
const YELLOW4 = colorant"#FFD43B"
const ORANGE4 = colorant"#FFA94D"

function plot_ARIMA_mean(means, tdir::Symbol, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    if tdir == :day
        x_means = 1:366*24
        _xlabel = "day * time"
    elseif tdir == :hour
        x_means = 0:23
        _xlabel = "time"
    else
        error("Time directive must be between :day, :hour")
    end

    arima_mean_path = output_dir * "$(output_prefix).png"
    pl = Plots.plot(x_means, means,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=_xlabel, ylabel=String(ycol), legend=false)
    Plots.png(pl, arima_mean_path)

    arima_mean_csvpath = output_dir * "$(output_prefix).csv"
    df = DataFrame(tdir => x_means, :mean => means)
    CSV.write(arima_mean_csvpath, df, writeheader = true)

    nothing
end

function plot_ARIMA_mean_smoothing(means, tdir::Symbol, ycol::Symbol, us, vs,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    if tdir == :day
        x_means = 1:366*24
        _xlabel = "day * time"
    elseif tdir == :hour
        x_means = 0:23
        _xlabel = "time"
    else
        error("Time directive must be between :day, :hour")
    end

    arima_fluc_path = output_dir * "$(output_prefix).png"
    pl = Plots.plot(x_means, means,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=_xlabel, ylabel=String(ycol), legend=false)
    Plots.plot!(us, vs, linewidth=2, color=:red)
    Plots.png(pl,arima_fluc_path)

    arima_fluc_csvpath = output_dir * "$(output_prefix).csv"
    df = DataFrame(tdir => x_means, :mean => means)
    CSV.write(arima_fluc_csvpath, df, writeheader = true)

    nothing
end

function plot_ARIMA_fluc(df::DataFrame, tdir::Symbol, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    arima_fluc_path = output_dir * "$(output_prefix).png"
    dates = DateTime.(df[!, :date], Local)
    pl = Plots.plot(dates, df[!, Symbol(tdir, "_fluc")],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel="dates", ylabel=String(ycol), legend=false)
    Plots.png(pl,arima_fluc_path)

    arima_fluc_csvpath = output_dir * "$(output_prefix).csv"
    df = DataFrame(:date => dates, :fluc => df[!, Symbol(tdir, "_fluc")])
    CSV.write(arima_fluc_csvpath, df, writeheader = true)

    nothing
end

function plot_ARIMA_acf(df::DataFrame, output_dir::String, output_prefix::String, sim_name::String)
    ENV["GKSwstype"] = "100"

    # ACF
    plot_path = output_dir * "$(output_prefix)_acf_$(sim_name).png"

    pl = Plots.plot(df[:, :lags], df[:, :acf],
        size = (2560, 1080),
        line=:solid, linewidth=5,
        ylim=(min(0.0, minimum(df[:, :acf])), 1.0),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=30PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color= :black,
        xlabel = "lags", ylabel = "acf", legend=false)

    Plots.png(pl, plot_path)

    # PACF
    plot_path = output_dir * "$(output_prefix)_pacf_$(sim_name).png"

    pl = Plots.plot(df[:, :lags], df[:, :pacf],
        size = (2560, 1080),
        line=:solid, linewidth=5,
        ylim=(min(0.0, minimum(df[:, :pacf])), 1.0),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=30PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color= :black,
        xlabel = "lags", ylabel = "pacf", legend=false)

    Plots.png(pl, plot_path)

    nothing
end

function plot_ARIMA_corr(df_corr::DataFrame, output_size::Integer, output_dir::String, output_prefix::String, sym_name::Symbol)

    ENV["GKSwstype"] = "100"

    plot_path = output_dir * "$(output_prefix)_corr_hourly_$(String(sym_name))"

    pl = Plots.plot(float.(df_corr[!, :hour]), float.(df_corr[!, :corr]),
        size = (2560, 1080),
        title="Correlation on hourly prediction", xlabel="hour", ylabel="corr",
        line=:solid, linewidth=5, label="OBS",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linecolor = LN01_COLOR, legend=false)

    png(pl, plot_path)

    nothing
end

function plot_ARIMA_scatter(dfs::Array{DataFrame, 1}, ycol::Symbol,
    output_size::Integer, output_dir::String, output_prefix::String, sym_name::Symbol)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        sc_path = output_dir * "$(i_pad)/" * "$(output_prefix)_scatter_$(String(sym_name))_$(i_pad)h.png"
        df = dfs[i]
        lim = max(maximum(float.(df[!, :y])), maximum(float.(df[!, :yhat])))

        sc = Plots.scatter(float.(df[!, :y]), float.(df[!, :yhat]),
            size = (1080, 1080),
            xlim = (0, lim), ylim = (0, lim), legend=false,
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            title="DNN/OBS ($(i_pad)h)", xlabel="OBS", ylabel="DNN",
            background_color = BG_COLOR, markercolor = MK_COLOR)

        # diagonal line (corr = 1)
        Plots.plot!(collect(0:0.1:lim), collect(0:0.1:lim),
            xlim = (0, lim), ylim = (0, lim), legend=false,
            background_color = BG_COLOR, linecolor = LN_COLOR)
        Plots.png(sc, sc_path)

        sc_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_scatter_$(i_pad)h.csv"
        outdf = DataFrame(obs = float.(df[!, :y]), model = float.(df[!, :yhat]))
        CSV.write(sc_csvpath, outdf, writeheader = true)
    end

    nothing
end

function plot_ARIMA_lineplot(dfs::Array{DataFrame, 1}, ycol::Symbol,
    output_size::Integer, test_fdate::DateTime, test_tdate::DateTime,
    output_dir::String, output_prefix::String, sym_name::Symbol)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        line_plotpath = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(String(sym_name))_$(i_pad)h.png"
        # filter by date because batched input makes `undef` DateTime values and throw Overflow
        df = filter(row -> test_fdate <= row[:date] <= test_tdate, dfs[i])
        dates_h = df[!, :date] .+ Dates.Hour(i)

        pl = Plots.plot(dates_h, float.(df[!, :y]),
            size = (2560, 1080),
            ylim = (0.0,
                max(maximum(float.(df[!, :y])), maximum(float.(df[!, :yhat])))),
            line=:solid, linewidth=5, label="OBS",
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            background_color = BG_COLOR, color=LN01_COLOR,
            title=String(ycol) * " by dates ($(i_pad)h)",
            xlabel="date", ylabel=String(ycol), legend=:best)

        pl = Plots.plot!(dates_h, float.(df[!, :yhat]),
            line=:solid, linewidth=5, color=LN02_COLOR, label="DNN")
        Plots.png(pl, line_plotpath)

        line_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.csv"
        outdf = DataFrame(dates = dates_h, obs = float.(df[!, :y]), model = float.(df[!, :yhat]))
        CSV.write(line_csvpath, outdf, writeheader = true)
    end

    nothing
end