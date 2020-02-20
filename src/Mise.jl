module Mise

# Base
using Base.Filesystem
using Base.Iterators: partition, zip
using LinearAlgebra
using Printf
using Random
using Statistics

# Math
using NumericalIntegration
using Dierckx

# Statistics
#using Distributions
using StatsBase: mean, std, mean_and_std, zscore, crosscor
import StatsBase: zscore!
using StatsBase
using KernelDensity
using NearestNeighbors

# Statistical Models
using TimeSeries
using HypothesisTests
using Loess
using StateSpaceModels

# ML
using Flux
using Zygote

# GPU
using CuArrays

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
using Colors, ColorTypes
using StatsPlots: @df, StatsPlots

if isa(Sys.which("python3"), String)
    using ExcelReaders
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

# Statistical Methods
include("analysis/analysis.jl")
include("analysis/smoothing.jl")
include("analysis/plot_analysis.jl")
include("ARIMA/ARIMA.jl")
include("ARIMA/plot_ARIMA.jl")
include("OU/model.jl")

# Machine Learning
include("DNN/preprocess.jl")
include("DNN/model.jl")
include("LSTNet/model.jl")

include("postprocess/post_DNN.jl")

# input
export join_data, filter_raw_data, filter_station, filter_jongro, read_jongro,
     parse_aerosols,
# utils
    extract_col_statvals, zscore!, unzscore, unzscore!,
    minmax_scaling, minmax_scaling!,
    unminmax_scaling, unminmax_scaling!,
    exclude_elem, findrow,
    WHO_PM10, WHO_PM25,
# data
    get_date_range, validate_dates, window_df,
    split_sizes3, split_sizes2,
    remove_sparse_input!, is_sparse_Y,
    getX, getY, make_pair_DNN, make_batch_DNN, 
    serializeBatch, unpack_seq, matrix2arrays,
    make_pair_LSTNet, make_batch_LSTNet,
    zero2Missing!, impute!,
# activation
# evaluation
    evaluations, evaluation,
    RMSE, MAE, MSPE, MAPE,
    RSR, NSE, PBIAS, IOA, RefinedIOA, R2, AdjR2, classification,
# loss
    huber_loss, huber_loss_mean, mse_rnn,
# smoothing
    periodic_mean, populate_periodic_mean,
    season_adj_lee, compute_annual_mean,
# analysis & plot
    compute_inttscale, mean_aucotor, pdf,
    plot_anal_lineplot, 
    plot_anal_pdf, plot_anal_autocor,
    plot_anal_correlogram, plot_anal_violin,
    plot_anal_periodic_mean, plot_anal_periodic_fluc,
# ARIMA
    smoothing_mean,
    plot_seasonality,
    plot_ARIMA_fluc, plot_ARIMA1_mean, plot_ARIMA_mean_smoothing, 
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
    predict_DNN_model_zscore, predict_DNN_model_minmax,
    predict_DNN_model_logzscore, predict_DNN_model_logminmax,
    predict_DNN_model_invzscore,
    predict_RNN_model_zscore, predict_RNN_model_minmax,
    export_CSV,
# plot
    plot_histogram,
    plot_lineplot_total,
    plot_corr_input,
    plot_pcorr,
    plot_corr,
    plot_DNN_scatter,
    plot_DNN_histogram,
    plot_DNN_lineplot
end
