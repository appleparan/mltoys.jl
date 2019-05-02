#using CuArrays
using CSV
using Dates, TimeZones
using DataFrames
using Flux
using Flux.Tracker
using JuliaDB

using Plots
using Plots.PlotMeasures
using UnicodePlots

ENV["GKSwstype"] = "100"

function plot_totaldata(df::DataFrame, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hist_path = output_dir * String(ycol) * "_total_hist.png"
    plot_path = output_dir * String(ycol) * "_total_plot.png"

    # plot histogram
    gr()
    ht = Plots.histogram(df[ycol], title="Histogram of " * String(ycol),
        xlabel=String(ycol), ylabel="# of data", bins=200,
        fillcolor=[:red], fillalpha=0.2, legend=false)
    png(ht, hist_path)

    # plot in dates
    dates = DateTime.(df[:date])
    gr(size = (2100, 900))
    pl = Plots.plot(dates, df[ycol], title=String(ycol) * " in dates", xlabel="date", ylabel=String(ycol), margin=15px, legend=false)
    png(pl, plot_path)
end

function plot_initdata(dataset, ycol::Symbol, μ::AbstractFloat, σ::AbstractFloat, output_dir::String)
    ENV["GKSwstype"] = "100"
    hist_path = output_dir * String(ycol) * "_total_hist.png"

    init_table = table((y = [],))
    for (x, y) in dataset

        cpu_y = y |> cpu
        org_y = cpu_y .* σ .+ μ

        tmp_table = table((y = org_y,))
        init_table = merge(init_table, tmp_table)
    end

    gr()
    ht = Plots.histogram(JuliaDB.select(init_table, :y), title="Histogram of initial data: " * String(ycol),
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        xlabel=String(ycol), ylabel="# of data", bins=200,
        fillalpha=0.5, legend=false)
    png(ht, hist_path)
end

function plot_DNN(df, dataset, model, ycol::Symbol, μ::AbstractFloat, σ::AbstractFloat, output_dir::String)
    ENV["GKSwstype"] = "100"
    sc_01h_path = output_dir * String(ycol) * "_01h_scatter.png"
    hs_01h_path = output_dir * String(ycol) * "_01h_hist.png"
    sc_24h_path = output_dir * String(ycol) * "_24h_scatter.png"
    hs_24h_path = output_dir * String(ycol) * "_24h_hist.png"
    line_01h_path = output_dir * String(ycol) * "_01h_line.png"
    line_24h_path = output_dir * String(ycol) * "_24h_line.png"
    table_01h_path = output_dir * String(ycol) * "_01h_table.csv"
    table_24h_path = output_dir * String(ycol) * "_01h_table.csv"

    dnn_01h_table = table((y = [], ŷ = [],))
    dnn_24h_table = table((y = [], ŷ = [],))

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ .+ μ
        
        tmp_table = table((y = org_y[1], ŷ = org_ŷ[1],))
        dnn_01h_table = merge(dnn_table, tmp_table)

        tmp_table = table((y = org_y[24], ŷ = org_ŷ[24],))
        dnn_24h_table = merge(dnn_table, tmp_table)
    end

    lim = max(maximum(JuliaDB.select(dnn_01h_table, :y)), maximum(JuliaDB.select(dnn_01h_table, :ŷ)))
    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ), 
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        xlim = (0, lim), ylim = (0, lim), legend=false,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN")
    plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false)
    png(sc, sc_01h_path)

    gr()
    gr(size=(2560, 1080))
    ht = Plots.histogram([JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ)],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        label=["obs", "model"],
        title="Histogram of data", ylabel="# of data", fillalpha=0.5)
    png(ht, hs_01h_path)

    lim = max(maximum(JuliaDB.select(dnn_24h_table, :y)), maximum(JuliaDB.select(dnn_24h_table, :ŷ)))
    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ), 
        guidefontsize = 24, titlefontsize = 40, tickfontsize = 18, 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN")
    plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false)
    png(sc, sc_24h_path)

    gr(size=(2560, 1080))
    ht = Plots.histogram([JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ)],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        label=["obs", "model"],
        title="Histogram of data", ylabel="# of data", fillalpha=0.5)
    png(ht, hs_24h_path)

    # plot in dates
    dates = DateTime.(df[:date])
    gr(size = (2100, 900))
    pl = Plots.plot(dates, [JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ)],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        line=[:dot, :solid], color=[:red, :black], label=["obs.", "model"],
        title=String(ycol) * " in dates (1h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    png(pl, line_01h_path)

    gr(size = (2100, 900))
    pl = Plots.plot(dates, [JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ)],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, margin=15px,
        line=[:dot, :solid], color=[:red, :black], label=["obs.", "model"],
        title=String(ycol) * " in dates (24h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    png(pl, line_24h_path)

    #save(String(ycol) * "_table.csv", dnn_table)
    JuliaDB.save(dnn_01h_table, table_01h_path)
    JuliaDB.save(dnn_24h_table, table_24h_path)
end
