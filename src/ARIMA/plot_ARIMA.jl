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
