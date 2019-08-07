module Mise

# Base
using Base.Filesystem
using Base.Iterators: partition, zip
using LinearAlgebra: norm
using Printf
using Random
using Statistics

# Statistics
using StatsBase: mean, std, mean_and_std, zscore, crosscor
import StatsBase: zscore!

# ML
using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data
# for temporal fix
using ForwardDiff

# Tables
using DataFrames, DataFramesMeta, Missings, Query
using DataValues
using Dates, TimeZones
using JuliaDB

# IO
using BSON, CSV
using DelimitedFiles
using FileIO
using Formatting
using Glob

# utils
using ArgParse
using MicroLogging
using ProgressMeter

# Plots
using Plots
using Plots.PlotMeasures
using ColorTypes
using StatsPlots

if isa(Sys.which("python3"), String)
        using ExcelReaders
end

if isa(Sys.which("nvcc"), String)
    using CuArrays
end

ENV["GKSwstype"] = "100"

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("utils.jl")
include("input.jl")
include("output.jl")
include("loss.jl")
include("evaluation.jl")
include("plots.jl")

include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")
#include("jongro02_LSTM/model.jl")

include("postprocess/post_jongro01_DNN.jl")

# temporal workaround for CuArrays #378
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float64,1}, ::Val{2}) = x*x

# input
export join_data, filter_jongro, read_jongro, filter_station, read_station,
# utils
        mean_and_std_cols, hampel!, zscore!, exclude_elem, split_df, window_df,
        split_sizes3, split_sizes2, create_chunks, create_idxs,
        getHoursLater, remove_sparse_input!, is_sparse_Y,
        getX, getY, make_pair_DNN, make_batch_DNN, 
        getX_LSTM, getY_LSTM, make_input_LSTM, findrow,
        WHO_PM10, WHO_PM25,
# evaluation
        evaluations, RMSE, RSR, NSE, PBIAS, IOA, classification,
# loss
        huber_loss, huber_loss_mean, mse_rnn,
# jongro01_DNN
        train_DNN, corr_input,
# jongro02_DNN
#       train_all_LSTM,
# post processing
        compute_corr, test_features, test_station, test_classification,
# output
        predict_model, export_CSV,
# plot
        plot_corr_input,
        plot_totaldata,
        plot_pcorr,
        plot_corr,
        plot_DNN_scatter,
        plot_DNN_histogram,
        plot_DNN_lineplot        
end
