using CuArrays
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

    gr(size = (800, 600))
    ht = Plots.histogram(df[ycol], title="Histogram of " * String(ycol), xlabel=String(ycol), ylabel="# of data", bins=200, legend=false)
    png(ht, hist_path)

    #date_fmt = Dates.DateFormat("yyyy-mm-dd HH:MM:SSz")
    dates = DateTime.(df[:date])
    gr(size = (2100, 900))
    pl = Plots.plot(dates, df[ycol], title=String(ycol) * " in dates", xlabel="date", ylabel=String(ycol), bottom_margin=15mm, legend=false)
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

    gr(size = (800, 600))
    ht = Plots.histogram(select(init_table, :y), title="Histogram of initial data: " * String(ycol), xlabel=String(ycol), ylabel="# of data", bins=200, legend=false)
    png(ht, hist_path)
end

function plot_DNN(dataset, model, ycol::Symbol, μ::AbstractFloat, σ::AbstractFloat, output_dir::String)
    ENV["GKSwstype"] = "100"
    sc_path = output_dir * String(ycol) * "_scatter.png"
    hs_path_org = output_dir * String(ycol) * "_org_hist.png"
    hs_path_mdl = output_dir * String(ycol) * "_dnn_hist.png"

    dnn_table = table((y = [], ŷ = [],))
    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ .+ μ
        
        tmp_table = table((y = org_y, ŷ = org_ŷ,))
        dnn_table = merge(dnn_table, tmp_table)
    end

    gr(size = (800, 600))
    sc = Plots.scatter(select(dnn_table, :y), select(dnn_table, :ŷ), title="OBS/DNN", xlabel="Observation", ylabel="DNN")
    png(sc, sc_path)

    gr(size = (800, 600))
    ht = Plots.histogram(select(dnn_table, :y), title="Histogram of original data ", ylabel="# of data", bins=200, legend=false)
    png(ht, hs_path_org)

    gr(size = (800, 600))
    ht = Plots.histogram(select(dnn_table, :ŷ), title="Histogram of modeled data ", ylabel="# of data", bins=200, legend=false)
    png(ht, hs_path_mdl)
end

function plot_DNN_toCSV(dataset, model, μ::AbstractFloat, σ::AbstractFloat, output_path::String)
    dnn_table = table(y = [], ŷ = [])
    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ_data .+ μ_data
        
        tmp_table = table(y = org_y, ŷ = org_ŷ)
        dnn_table = merge(dnn_table, tmp_table)
    end

    JuliaDB.save(dnn_table, output_path)
end