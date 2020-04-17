function predict_DNN_model_zscore(dataset::AbstractVector{Tuple{V1, V1, V2}}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where
        {V1<:AbstractVector{<:Real}, V2<:AbstractVector{<:DateTime}}

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = unzscore(cpu_y, μ, σ)
        # 24 hour prediction
        org_ŷ = unzscore(cpu_ŷ, μ, σ)

        dnn_res_df_tmp = DataFrame(date = dates,
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_DNN_model_zscore(dataset::AbstractVector{Tuple{M1, M1, M2}}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where
        {M1<:AbstractMatrix{<:Real}, M2<:AbstractMatrix{<:DateTime}}

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = unzscore(cpu_y, μ, σ)
        # 24 hour prediction
        org_ŷ = unzscore(cpu_ŷ, μ, σ)

        # size(x, 2) == size(y, 2) == size(dates, 2) == batch_size
        for i = 1:size(y, 2)
            dnn_res_df_tmp = DataFrame(date = dates,
                offset = 1:output_size,
                y = org_y[:, i],
                yhat = org_ŷ[:, i])
            append!(dnn_res_df, dnn_res_df_tmp)
        end
    end

    dnn_res_df
end

function predict_DNN_model_zscore_season(dataset::AbstractVector{T}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat, season_table::AbstractNDSparse,
    _eltype::DataType, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = compose_seasonality(collect(DateTime.(dates, Local)),
            unzscore(cpu_y, μ, σ),
            season_table)

        # 24 hour prediction
        org_ŷ = compose_seasonality(collect(DateTime.(dates, Local)),
            unzscore(cpu_ŷ, μ, σ),
            season_table)

        dnn_res_df_tmp = DataFrame(date = DateTime.(dates, Local),
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_DNN_model_minmax(dataset::AbstractVector{T}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = unminmax_scaling(cpu_y, _min, _max, a, b)
        # 24 hour prediction
        org_ŷ = unminmax_scaling(cpu_ŷ, _min, _max, a, b)

        dnn_res_df_tmp = DataFrame(date = DateTime.(dates, Local),
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_DNN_model_logzscore(dataset::AbstractVector{T}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = exp.(unzscore(cpu_y, μ, σ)) .- 10.0
        # 24 hour prediction
        org_ŷ = exp.(unzscore(cpu_ŷ, μ, σ)) .- 10.0

        dnn_res_df_tmp = DataFrame(date = DateTime.(dates, Local),
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_DNN_model_invzscore(dataset::AbstractVector{T}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = 1.0 ./ (unzscore(cpu_y, μ, σ)) .- 10.0
        # 24 hour prediction
        org_ŷ = 1.0 ./ (unzscore(cpu_ŷ, μ, σ)) .- 10.0

        dnn_res_df_tmp = DataFrame(date = DateTime.(dates, Local),
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_DNN_model_logminmax(dataset::AbstractVector{T}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
     _eltype::DataType, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (x, y, dates) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = exp.(unminmax_scaling(cpu_y, _min, _max, a, b)) .- 10.0
        # 24 hour prediction
        org_ŷ = exp.(unminmax_scaling(cpu_ŷ, _min, _max, a, b)) .- 10.0

        dnn_res_df_tmp = DataFrame(date = DateTime.(dates, Local),
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(dnn_res_df, dnn_res_df_tmp)
    end

    dnn_res_df
end

function predict_RNN_model_zscore(dataset::AbstractVector{Tuple{F4, AF2, F2, D1}}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
     _eltype::DataType, output_size::Integer, output_dir::String) where
        {F4<:AbstractArray{<:Real, 4}, F2<:AbstractArray{<:Real, 2},
        AF2<:AbstractVector{<:AbstractArray{<:Real, 2}}, D1<:AbstractVector{<:DateTime}}
    # pair test set (xe, xd, y) is same as batch with batch_size == 1, but `dates` are Vector
    rnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (xe, xd, y, dates) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y[:, 1] |> cpu
        cpu_ŷ = ŷ[:, 1] |> cpu

        # 24 hour data
        org_y = unzscore(cpu_y, μ, σ)
        # 24 hour prediction
        org_ŷ = unzscore(cpu_ŷ, μ, σ)

        rnn_res_df_tmp = DataFrame(date = dates,
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(rnn_res_df, rnn_res_df_tmp)
    end

    rnn_res_df
end

function predict_RNN_model_zscore(dataset::AbstractVector{Tuple{F4, AF2, F2, D2}}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat,
     _eltype::DataType, output_size::Integer, output_dir::String) where
        {F4<:AbstractArray{<:Real, 4}, F2<:AbstractArray{<:Real, 2},
        AF2<:AbstractVector{<:AbstractArray{<:Real, 2}}, D2<:AbstractMatrix{<:DateTime}}

    # batch test set, only difference with pair test set is type of `dates`
    rnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    for (xe, xd, y, dates) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # size(x, 2) == size(y, 2) == size(dates, 2) == batch_size
        for i = 1:size(y, 2)
            # 24 hour data
            org_y = unzscore(cpu_y, μ, σ)
            # 24 hour prediction
            org_ŷ = unzscore(cpu_ŷ, μ, σ)

            rnn_res_df_tmp = DataFrame(date = dates[:, i],
                offset = 1:output_size,
                y = org_y[:, i],
                yhat = org_ŷ[:, i])
            append!(rnn_res_df, rnn_res_df_tmp)
        end
    end

    rnn_res_df
end

function predict_RNN_model_minmax(dataset::AbstractVector{Tuple{V1, V1, V1, V2}}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where
        {V1<:AbstractVector{<:Real}, V2<:AbstractVector{<:DateTime}}

    rnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (xe, xd, y, dates) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y[:, 1] |> cpu
        cpu_ŷ = ŷ[:, 1] |> cpu

        # 24 hour data
        org_y = unminmax_scaling(cpu_y, _min, _max, a, b)
        # 24 hour prediction
        org_ŷ = unminmax_scaling(cpu_ŷ, _min, _max, a, b)

        rnn_res_df_tmp = DataFrame(date = dates,
            offset = 1:output_size,
            y = org_y,
            yhat = org_ŷ)
        append!(rnn_res_df, rnn_res_df_tmp)
    end

    rnn_res_df
end

function predict_RNN_model_minmax(dataset::AbstractVector{Tuple{M1, M1, M1, M2}}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    _eltype::DataType, output_size::Integer, output_dir::String) where
        {M1<:AbstractMatrix{<:Real}, M2<:AbstractMatrix{<:DateTime}}

    rnn_res_df = DataFrame(date = DateTime[], offset = Int[], y = _eltype[], yhat = _eltype[])

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (xe, xd, y, dates) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y[:, 1] |> cpu
        cpu_ŷ = ŷ[:, 1] |> cpu

        # 24 hour data
        org_y = unminmax_scaling(cpu_y, _min, _max, a, b)
        # 24 hour prediction
        org_ŷ = unminmax_scaling(cpu_ŷ, _min, _max, a, b)

        # size(x, 2) == size(y, 2) == size(dates, 2) == batch_size
        for i = 1:size(x, 2)
            rnn_res_df_tmp = DataFrame(date = dates[:, i],
                offset = 1:output_size,
                y = org_y[:, i],
                yhat = org_ŷ[:, i])
            append!(rnn_res_df, rnn_res_df_tmp)
        end
    end

    rnn_res_df
end

"""
    export_CSV

Basic NDSparse
"""
function export_CSV(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1},
    ycol::Symbol, output_size::Integer,
    _eltype::DataType, output_dir::String, output_prefix::String)

    dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath(output_dir * "$(i_pad)/")
        plottable_path::String = output_dir * "$(i_pad)/" * "$(output_prefix)_plottable_$(i_pad)h.csv"

        dates_h = dates .+ Dates.Hour(i)
        y = JuliaDB.select(dnn_table[i], :y)
        ŷ = JuliaDB.select(dnn_table[i], :ŷ)

        len_data = min(length(dates), length(y), length(ŷ))

        df = DataFrame(
            date = dates_h[1:len_data], y = y[1:len_data], yhat = ŷ[1:len_data])

        dfs[i] = df

        CSV.write(plottable_path, df)
    end

    dfs
end

"""
    export_CSV

with seasonality decomposition
"""
function export_CSV(dates::Array{DateTime, 1}, table::AbstractNDSparse, season_table::AbstractNDSparse,
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

    dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath(output_dir * "$(i_pad)/")
        plotdf_path::String = output_dir * "$(i_pad)/" * "$(output_prefix)_plottable_$(i_pad)h.csv"

        dates_h = dates .+ Dates.Hour(i)
        y = JuliaDB.select(table[i], :y)
        ŷ = compose_seasonality(dates_h, JuliaDB.select(table[i], :ŷ), season_table)

        len_data = min(length(dates), length(y), length(ŷ))

        df = DataFrame(
            date = dates_h[1:len_data], y = y[1:len_data], yhat = ŷ[1:len_data])

        sort!(df, (:date))
        dfs[i] = df

        CSV.write(plotdf_path, df)
    end

    dfs
end


"""
    export_CSV

new DataFrame
"""
function export_CSV(dates::Array{DateTime, 1}, df::DataFrame,
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

    offset_dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath(output_dir * "$(i_pad)/")
        plotdf_path::String = output_dir * "$(i_pad)/" * "$(output_prefix)_plottable_$(i_pad)h.csv"

        dates_h = dates .+ Dates.Hour(i)
        offset_df = filter(row -> row[:offset] == i, df)

        sort!(offset_df, (:date))
        offset_dfs[i] = offset_df

        CSV.write(plotdf_path, offset_df)
    end

    offset_dfs
end

"""
    export_CSV

new DataFrame + seasonality
"""
function export_CSV(dates::Array{DateTime, 1}, df::DataFrame, season_table::AbstractNDSparse,
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

    offset_dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath(output_dir * "$(i_pad)/")
        plotdf_path::String = output_dir * "$(i_pad)/" * "$(output_prefix)_plottable_$(i_pad)h.csv"

        dates_h = dates .+ Dates.Hour(i)
        offset_df = filter(row -> row[:offset] == i, df)

        sort!(offset_df, (:date))
        offset_dfs[i] = offset_df

        CSV.write(plotdf_path, offset_df)
    end

    offset_dfs
end

function compute_corr(dnn_table::Array{IndexedTable, 1},
    output_size::Integer, output_dir::String, output_prefix::String)

    corr_path = output_dir * output_prefix * "_corr.csv"
    corr = zeros(output_size)

    for i = 1:output_size
        corr[i] = Statistics.cor(JuliaDB.select(dnn_table[i], :y), JuliaDB.select(dnn_table[i], :ŷ))
    end

    df = DataFrame(hour = collect(1:output_size), corr = corr)

    CSV.write(corr_path, df)

    df
end

function compute_corr(dfs::Array{DataFrame, 1},
    output_size::Integer, output_dir::String, output_prefix::String)

    corr_path = output_dir * output_prefix * "_corr.csv"
    corr = zeros(output_size)

    for i = 1:output_size
        _df = dfs[i]
        corr[i] = Statistics.cor(_df[!, :y], _df[!, :yhat])
    end

    df = DataFrame(hour = collect(1:output_size), corr = corr)

    CSV.write(corr_path, df)

    df
end
