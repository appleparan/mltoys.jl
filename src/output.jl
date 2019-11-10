function predict_model_zscore(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
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

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ .+ μ

        for i = 1:output_size
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function predict_model_minmax(dataset::AbstractArray{T, 1}, model, ycol::Symbol,
    _min::AbstractFloat, _max::AbstractFloat, a::Real, b::Real,
    output_size::Integer, output_dir::String) where T <: Tuple

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
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        org_y = (cpu_y .- a) .* (_max - _min) .* (b - a) .+ _min
        org_ŷ = (Flux.Tracker.data(cpu_ŷ) .- a) .* (_max - _min) .* (b - a) .+ _min

        for i = 1:output_size
            tmp_table = table((y = [org_y[i]], ŷ = [org_ŷ[i]],))
            dnn_table[i] = merge(dnn_table[i], tmp_table)
        end
    end

    dnn_table
end

function export_CSV(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1},
    ycol::Symbol, output_size::Integer, output_dir::String, output_prefix::String)

    dfs = Array{DataFrame}(undef, output_size)

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
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
