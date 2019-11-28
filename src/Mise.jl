module Mise

# Base
using Base.Filesystem
using Base.Iterators: partition, zip
using LinearAlgebra
using Printf
using Random
using Statistics

# Statistics
#using Distributions
using StatsBase: mean, std, mean_and_std, zscore, crosscor
import StatsBase: zscore!

# ML
using Flux
# for temporal fix
using ForwardDiff

# MCMC
#using Mamba

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
    # temporal workaround for CuArrays #378
    CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x
    CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float64,1}, ::Val{2}) = x*x
end

ENV["GKSwstype"] = "100"

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("utils.jl")
include("data.jl")
include("input.jl")
include("output.jl")
include("loss.jl")
include("activation.jl")
include("evaluation.jl")
include("plots.jl")

include("DNN/preprocess.jl")
include("DNN/model.jl")
include("OU/model.jl")
include("LSTNet/model.jl")

include("postprocess/post_DNN.jl")

# input
export join_data, filter_raw_data, filter_jongro, read_jongro,
# utils
    extract_col_statvals, zscore!, min_max_scaling!,
    exclude_elem, findrow,
    WHO_PM10, WHO_PM25,
# data
    get_date_range, validate_dates, window_df,
    split_sizes3, split_sizes2,
    remove_sparse_input!, is_sparse_Y,
    getX, getY, make_pair_DNN, make_batch_DNN, 
    serializeBatch, unpack_seq, matrix2arrays,
    make_pair_LSTNet, make_batch_LSTNet,
# activation
# evaluation
    evaluations, RMSE, MAE, RSR, NSE, PBIAS,
    IOA, RefinedIOA, R2, AdjR2, MSPE, MAPE, classification,
# loss
    huber_loss, huber_loss_mean, mse_rnn,
# DNN
    train_DNN, corr_input,
# preprocess
    load_data_DNN, filter_station_DNN, process_raw_data_DNN!, read_station,
# LSTNet
    train_LSTNet,
# OU
    evolve_OU,
# post processing
    compute_corr, test_features, test_station, test_classification,
# output
    predict_DNNmodel_zscore, predict_DNNmodel_minmax,
    predict_RNNmodel_zscore, predict_RNNmodel_minmax,
    export_CSV,
# plot
    plot_corr_input,
    plot_totaldata,
    plot_pcorr,
    plot_corr,
    plot_DNN_scatter,
    plot_DNN_histogram,
    plot_DNN_lineplot
end
