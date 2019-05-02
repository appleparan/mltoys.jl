#using CuArrays
using CSV
using Dates, TimeZones
using DataFrames
using Flux
using Flux.Tracker
using JuliaDB

using Plots
using Plots.PlotMeasures
using ColorTypes

ENV["GKSwstype"] = "100"

const BG_COLOR = RGB(248/255, 247/255, 247/255)
const LN_COLOR = RGB(56/255, 44/255, 80/255)
const MK_COLOR = RGB(109/255, 117/255, 126/255)

function plot_totaldata(df::DataFrame, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hist_path = output_dir * String(ycol) * "_total_hist.png"
    plot_path = output_dir * String(ycol) * "_total_plot.png"

    # plot histogram
    gr(size=(1920, 1080))
    ht = Plots.histogram(df[ycol], title="Histogram of " * String(ycol),
        xlabel=String(ycol), ylabel="#", bins=200,
        margin=15px, legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, 
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, fillcolor=[:red], fillalpha=0.2)
    png(ht, hist_path)

    # plot in dates
    dates = DateTime.(df[:date])
    gr(size=(1920, 1080))
    pl = Plots.plot(dates, df[ycol],
        title=String(ycol) * " in dates", xlabel="date", ylabel=String(ycol),
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, 
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, background_color = BG_COLOR, linecolor = LN_COLOR, legend=false)
    png(pl, plot_path)
end

function get_prediction_table(df, dataset, model, ycol::Symbol, μ::AbstractFloat, σ::AbstractFloat, output_dir::String)
    table_01h_path = output_dir * String(ycol) * "_01h_table.csv"
    table_24h_path = output_dir * String(ycol) * "_24h_table.csv"

    dnn_01h_table = table((y = [], ŷ = [],))
    dnn_24h_table = table((y = [], ŷ = [],))

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ .+ μ
        
        tmp_table = table((y = [org_y[1]], ŷ = [org_ŷ[1]],))
        dnn_01h_table = merge(dnn_01h_table, tmp_table)

        tmp_table = table((y = [org_y[24]], ŷ = [org_ŷ[24]],))
        dnn_24h_table = merge(dnn_24h_table, tmp_table)
    end

    JuliaDB.save(dnn_01h_table, table_01h_path)
    JuliaDB.save(dnn_24h_table, table_24h_path)

    dnn_01h_table, dnn_24h_table
end

function plot_DNN_scatter(dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    sc_01h_path = output_dir * String(ycol) * "_01h_scatter.png"
    sc_24h_path = output_dir * String(ycol) * "_24h_scatter.png"

    lim = max(maximum(JuliaDB.select(dnn_01h_table, :y)), maximum(JuliaDB.select(dnn_01h_table, :ŷ)))
    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ), 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN",
        margin=15px, background_color = BG_COLOR, markercolor = MK_COLOR)
    plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        background_color = BG_COLOR, linecolor = LN_COLOR)
    png(sc, sc_01h_path)

    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ), 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN",
        margin=15px, background_color = BG_COLOR, markercolor = MK_COLOR)
    plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        background_color = BG_COLOR, linecolor = LN_COLOR)
    png(sc, sc_24h_path)   
end

function plot_DNN_histogram(dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hs_01h_path = output_dir * String(ycol) * "_01h_hist.png"
    hs_24h_path = output_dir * String(ycol) * "_24h_hist.png"

    gr(size=(2560, 1080))
    ht = Plots.histogram(JuliaDB.select(dnn_01h_table, :y), label="obs", 
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, title="Histogram of data", ylabel="#",
        background_color = BG_COLOR, fillalpha=0.5)
    ht = Plots.histogram!(JuliaDB.select(dnn_01h_table, :ŷ), label="model")
    png(ht, hs_01h_path)

    gr(size=(2560, 1080))
    ht = Plots.histogram(JuliaDB.select(dnn_24h_table, :y), label=["obs", "model"],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, title="Histogram of data", ylabel="#",
        background_color = BG_COLOR, fillalpha=0.5)
    ht = Plots.histogram!(JuliaDB.select(dnn_24h_table, :ŷ), label="model")
    png(ht, hs_24h_path)
end

function plot_DNN_lineplot(dates, dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    line_01h_path = output_dir * String(ycol) * "_01h_line.png"
    line_24h_path = output_dir * String(ycol) * "_24h_line.png"
    dates_01h = dates .+ Dates.Hour(1)
    dates_24h = dates .+ Dates.Hour(24)
    len_model = length(JuliaDB.select(dnn_01h_table, :y))

    # plot in dates
    gr(size = (2560, 1080))
    pl = Plots.plot(dates_01h[1:len_model], JuliaDB.select(dnn_01h_table, :y),
        line=:dot, color=:red, label="obs.",
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (1h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_01h[1:len_model], JuliaDB.select(dnn_01h_table, :ŷ),
        line=:solid, color=:black, label="model")
    png(pl, line_01h_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(dates_24h[1:len_model], JuliaDB.select(dnn_24h_table, :y),
        line=:dot, color=:red, label="obs.",
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (24h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_24h[1:len_model], JuliaDB.select(dnn_24h_table, :ŷ),
        line=:solid, color=:black, label="model")
    png(pl, line_24h_path)
end
