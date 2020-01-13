const BG_COLOR = RGB(255/255, 255/255, 255/255)
const LN_COLOR = RGB(67/255, 75/255, 86/255)
const MK_COLOR = RGB(67/255, 75/255, 86/255)
const LN01_COLOR = RGB(202/255,0/255,32/255)
const LN02_COLOR = RGB(5/255,113/255,176/255)
const FL01_COLOR = RGB(239/255, 138/255, 98/255)
const FL02_COLOR = RGB(103/255, 169/255, 207/255)

function plot_anal_lineplot(dates::Array{DateTime, 1}, arr::AbstractArray, ycol::Symbol,
    order::Integer, lag::Integer,
    output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix)_anal_timediff_o$(order)l$(lag).png"

    pl = Plots.plot(dates, arr,
        size = (2560, 1080),   
        line=:solid, linewidth=5,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title="Time difference with lag: $(lag), order: $(order)",
        xlabel="date", ylabel=String(ycol), legend=false)
    Plots.png(pl, line_path)

    line_csvpath = output_dir * "$(output_prefix)_anal_timediff_o$(order)l$(lag).csv"
    df = DataFrame(date = dates,
        arr = arr)
    CSV.write(line_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_autocor(arr::AbstractArray, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix)_autocor.png"

    pl = Plots.plot(0:length(arr)-1, arr,
        size = (1920, 1080),
        line=:solid, linewidth=5,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, line_path)

    line_csvpath = output_dir * "$(output_prefix)_autocor.csv"
    df = DataFrame(arr = arr)
    CSV.write(line_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_correlogram(arr::AbstractArray, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    bar_path = output_dir * "$(output_prefix)_correlogram_bar.png"
    pl = Plots.bar(0:length(arr)-1, arr,
        size = (1920, 1080),
        bar_width = 0.5,
        markershape = :circle, markersize = 5, markercolor = LN_COLOR,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, bar_path)

    line_path = output_dir * "$(output_prefix)_correlogram_line.png"
    pl = Plots.plot(0:length(arr)-1, arr,
        size = (1920, 1080),
        bar_width = 0.5,
        markershape = :circle, markersize = 5, markercolor = LN_COLOR,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, line_path)

    bar_csvpath = output_dir * "$(output_prefix)_correlogram.csv"
    df = DataFrame(arr = arr)
    CSV.write(bar_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_violin(df::DataFrame, ycol::Symbol, tdir::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    violin_path = output_dir * "$(output_prefix)_violin.png"

    pl = StatsPlots.@df df violin(df[!, tdir], df[!, ycol],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(0.2, :blue, stroke(0)), legend=false)
    Plots.png(pl, violin_path)

    violin_csvpath = output_dir * "$(output_prefix)_violin.csv"
    df = DataFrame(tdir => df[!, tdir], :arr => df[!, ycol])
    CSV.write(violin_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_violin(df::DataFrame, ycol::Symbol, tdir::Symbol, means::AbstractArray,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    violin_path = output_dir * "$(output_prefix)_violin_$(string(tdir))ly.png"

    pl = StatsPlots.@df df violin(df[!, tdir], df[!, ycol],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(0.2, :blue, stroke(0)), legend=false)
    if tdir == :hour
        x_means = 0:23
    elseif tdir == :day
        x_means = 1:366
    elseif tdir == :month
        x_means = 1:12
    elseif tdir == :quarter
        x_means = 1:4
    else
        error("Time directive must be between :hour, :day, :month, :quarter")
    end
    Plots.plot!(x_means, means, color=:red, marker=(2.0, :red, stroke(1)), legend=:false)
    Plots.png(pl, violin_path)

    violin_csvpath = output_dir * "$(output_prefix)_violin_$(string(tdir))ly.csv"
    df = DataFrame(tdir => df[!, tdir], :arr => df[!, ycol])
    CSV.write(violin_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_pdf(k::UnivariateKDE, ycol::Symbol, 
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    pdf_path = output_dir * "$(output_prefix).png"

    pl = Plots.plot(k.x, k.density,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel="hour", ylabel=String(ycol), 
        marker=(1, :blue, stroke(0)), legend=false)
    Plots.png(pl, pdf_path)

    pdf_df = DataFrame(x = k.x, density = k.density)
    pdf_csvpath = output_dir * "$(output_prefix).csv"    
    CSV.write(pdf_csvpath, pdf_df, writeheader = true)

    nothing
end

function plot_anal_time_mean(df::DataFrame, ycol::Symbol, tdir::Symbol, means::AbstractArray,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    time_mean_path = output_dir * "$(output_prefix)_$(string(tdir))ly.png"

    if tdir == :hour
        x_means = 0:23
    elseif tdir == :day
        x_means = 1:366
    elseif tdir == :month
        x_means = 1:12
    elseif tdir == :quarter
        x_means = 1:4
    else
        error("Time directive must be between :hour, :day, :month, :quarter")
    end

    pl = Plots.plot(x_means, means,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(1.0, :blue, stroke(0)), legend=false)
    Plots.png(pl, time_mean_path)

    time_mean_csvpath = output_dir * "$(output_prefix)_$(string(tdir))ly.csv"
    df = DataFrame(tdir => x_means, :mean => means)
    CSV.write(time_mean_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_time_fluc(df::DataFrame, tdir::Symbol, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    time_fluc_path = output_dir * "$(output_prefix)_time_fluc_$(string(tdir))ly.png"

    pl = Plots.plot(df[!, tdir], df[!, :fluc],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(1.0, :blue, stroke(0)), legend=false)
    Plots.png(pl, time_fluc_path)

    time_fluc_csvpath = output_dir * "$(output_prefix)_time_fluc_$(string(tdir))ly.csv"
    df = DataFrame(tdir => df[!, tdir], :fluc => df[!, :fluc])
    CSV.write(time_fluc_csvpath, df, writeheader = true)

    nothing
end
