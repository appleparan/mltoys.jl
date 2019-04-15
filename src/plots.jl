using CuArrays
using CSV
using DataFrames
using Flux
using Flux.Tracker

using Plots
using UnicodePlots

function plot_DNN(dataset, model, μ_data::Float64, σ_data::Float64, output_path::String)
    df = DataFrame(y = [], ŷ = [])
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        cpu_ŷ = ŷ |> cpu
        cpu_y = y |> cpu
        org_y = cpu_y .* σ_data .+ μ_data
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ_data .+ μ_data
        
        df_tmp = DataFrame(y = org_y, ŷ = org_ŷ)
        append!(df, df_tmp)
    end

    gr(size = (600, 500))
    Plots.scatter(df[:y], df[:ŷ], title="OBS/DNN", xlabel="Observation", ylabel="DNN")
    png(output_path)
    
    #UnicodePlots.scatterplot(df[:y], df[:ŷ], title="OBS/DNN", xlabel="Observation", ylabel="DNN")
    # StatsPlots
    # @df df StatsPlots.scatter(:y, :ŷ)
end

function plot_DNN_toCSV(dataset, model, μ_data::Float64, σ_data::Float64, output_path::String)
    df = DataFrame(y = [], ŷ = [])
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        cpu_ŷ = ŷ |> cpu
        cpu_y = y |> cpu
        org_y = cpu_y .* σ_data .+ μ_data
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ_data .+ μ_data
        
        df_tmp = DataFrame(y = org_y, ŷ = org_ŷ)
        append!(df, df_tmp)
    end

    CSV.write(output_path, df)
end