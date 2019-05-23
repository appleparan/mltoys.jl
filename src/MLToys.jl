module MLToys

using Base.Filesystem
using Base.Iterators: partition, zip
using LinearAlgebra: norm
using Printf
using Random
using Statistics

using BSON: @save, @load
using CSV
using DataFrames, Missings, Query
using DataValues
using Dates, TimeZones

using FileIO
using Glob
using JuliaDB
using MicroLogging
using ProgressMeter
using StatsBase: mean, std, mean_and_std, zscore

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

using Plots
using Plots.PlotMeasures
using ColorTypes
using StatsPlots

import StatsBase: zscore!

if isa(Sys.which("python3"), String)
        using ExcelReaders
end

if isa(Sys.which("nvcc"), String)
    using CuArrays
end

ENV["GKSwstype"] = "100"

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("input.jl")
include("utils.jl")
include("loss.jl")
include("evaluation.jl")
include("plots.jl")

include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")
include("jongro02_LSTM/model.jl")

# input
export join_data,
# utils
        mean_and_std_cols, hampel!, zscore!, exclude_elem, split_df, window_df,
        split_sizes3, split_sizes2, create_chunks, create_idxs,
        getHoursLater, remove_missing_pairs!, is_sparse_Y,
        getX_DNN, getY_DNN, make_pairs_DNN, make_minibatch_DNN, 
        getX_LSTM, getY_LSTM, make_input_LSTM,
# evaluation
        evaluations, RMSE, RSR, NSE, PBIAS, IOA,
# loss
        huber_loss, huber_loss_mean, mse_rnn,
# jongro01_DNN
        train_all_DNN, filter_jongro, read_jongro,
# jongro01_DNN
        train_all_LSTM,
# plot
        plot_totaldata,
        get_prediction_table,
        plot_corr,
        plot_DNN_scatter,
        plot_DNN_histogram,
        plot_DNN_lineplot        
end