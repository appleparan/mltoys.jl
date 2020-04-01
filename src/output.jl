function predict_DNN_model_zscore(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        dnn_table[i] = table((y = [], ŷ = [],))
    end

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = unzscore(cpu_y, μ, σ)
        # 24 hour prediction
        org_ŷ = unzscore(cpu_ŷ, μ, σ)

        for i = 1:output_size
            # 1 hour 
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_DNN_model_minmax(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    output_size::Integer, output_dir::String) where T <: Tuple

    dnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        dnn_table[i] = table((y = [], ŷ = [],))
    end

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = unminmax_scaling(cpu_y, _min, _max, a, b)
        # 24 hour prediction
        org_ŷ = unminmax_scaling(cpu_ŷ, _min, _max, a, b)

        for i = 1:output_size
            # 1 hour
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_DNN_model_logzscore(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        dnn_table[i] = table((y = [], ŷ = [],))
    end

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = exp.(unzscore(cpu_y, μ, σ)) .- 10.0
        # 24 hour prediction
        org_ŷ = exp.(unzscore(cpu_ŷ, μ, σ)) .- 10.0

        for i = 1:output_size
            # 1 hour 
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_DNN_model_invzscore(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat, output_size::Integer, output_dir::String) where T <: Tuple

    dnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        dnn_table[i] = table((y = [], ŷ = [],))
    end

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = 1.0 ./ (unzscore(cpu_y, μ, σ)) .- 10.0
        # 24 hour prediction
        org_ŷ = 1.0 ./ (unzscore(cpu_ŷ, μ, σ)) .- 10.0

        for i = 1:output_size
            # 1 hour
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_DNN_model_logminmax(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    output_size::Integer, output_dir::String) where T <: Tuple

    dnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        dnn_table[i] = table((y = [], ŷ = [],))
    end

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        # 24 hour data
        org_y = exp.(unminmax_scaling(cpu_y, _min, _max, a, b)) .- 10.0
        # 24 hour prediction
        org_ŷ = exp.(unminmax_scaling(cpu_ŷ, _min, _max, a, b)) .- 10.0

        for i = 1:output_size
            # 1 hour
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_RNN_model_zscore(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    μ::AbstractFloat, σ::AbstractFloat, output_size::Integer, output_dir::String) where T <: Tuple

    rnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        rnn_table[i] = table((y = [], ŷ = [],))
    end

    for (xe, xd, y) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y[:, 1] |> cpu
        cpu_ŷ = ŷ[:, 1] |> cpu

        # 24 hour data
        org_y = unzscore(cpu_y, μ, σ)
        # 24 hour prediction
        org_ŷ = unzscore(cpu_ŷ, μ, σ)

        for i = 1:output_size
            # 1 hour
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            rnn_table[i] = merge(rnn_table[i], tmp_table)
        end
    end

    rnn_table
end

function predict_RNN_model_minmax(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::AbstractFloat, b::AbstractFloat,
    output_size::Integer, output_dir::String) where T <: Tuple

    rnn_table = Array{IndexedTable}(undef, output_size)
    table_path = Array{String}(undef, output_size)

    for i = 1:output_size
        table_path[i] = output_dir * "$(String(ycol))_$(lpad(i,2,'0'))_table.csv"
        rnn_table[i] = table((y = [], ŷ = [],))
    end

    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    c = ((_max - _min) / (b - a))
    for (xe, xd, y) in dataset
        ŷ = model(xe |> gpu, xd |> gpu)

        cpu_y = y[:, 1] |> cpu
        cpu_ŷ = ŷ[:, 1] |> cpu

        # 24 hour data
        org_y = unminmax_scaling(cpu_y, _min, _max, a, b)
        # 24 hour prediction
        org_ŷ = unminmax_scaling(cpu_ŷ, _min, _max, a, b)

        for i = 1:output_size
            # 1 hour
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            rnn_table[i] = merge(rnn_table[i], tmp_table)
        end
    end

    rnn_table
end

function export_CSV(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1},
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

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

function export_CSV(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1}, season_table::AbstractNDSparse,
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

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

function export_CSV(dates::Array{DateTime, 1}, y::Array{Float64, 1}, ŷ::Array{Float64, 1},
    season_table::AbstractNDSparse,
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

    dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        plottable_path::String = output_dir * "$(i_pad)/" * "$(output_prefix)_plottable_$(i_pad)h.csv"

        dates_h = dates .+ Dates.Hour(i)
        y = JuliaDB.select(dnn_table[i], :y)
        ŷ = compose_seasonality(dates_h, JuliaDB.select(dnn_table[i], :ŷ), season_table)

        len_data = min(length(dates), length(y), length(ŷ))

        df = DataFrame(
            date = dates_h[1:len_data], y = y[1:len_data], yhat = ŷ[1:len_data])

        dfs[i] = df

        CSV.write(plottable_path, df)
    end

    dfs
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
