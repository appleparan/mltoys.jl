module MLToys

using Base.Filesystem
using Base.Iterators: partition
using Test
using LinearAlgebra: norm
using Printf
using Random
using Statistics

using BSON: @save, @load
using CSV
using DataFrames, Missings, Query
using DataValues
using Dates, TimeZones
using ExcelReaders
using FileIO
using Glob
using JuliaDB
using MicroLogging
using ProgressMeter
using StatsBase: mean_and_std, zscore, mean, std

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

using Plots
using Plots.PlotMeasures
using ColorTypes
using StatsPlots

if isa(Sys.which("nvcc"), String)
    using CuArrays
end

ENV["GKSwstype"] = "100"

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("input.jl")
include("utils.jl")
include("plots.jl")
include("evaluation.jl")

include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")
include("jongro02_LSTM/model.jl")

# input
export join_data, 
# utils
        mean_and_std_cols, hampel!, zscore!, exclude_elem, split_df, window_df,
        split_sizes3, split_sizes2, create_chunks, create_idxs,
        getHoursLater, getX, getY, make_pairs, make_minibatch, remove_missing_pairs!,
        getX_LSTM, getY_LSTM, make_pairs_LSTM, make_minibatch_LSTM,
# evaluation
        RSME, RSR, huber_loss, huber_loss_mean,
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