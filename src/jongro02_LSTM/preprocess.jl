using Random

using CSV
using DataFrames, Query, Missings
using Dates, TimeZones
using MicroLogging
using StatsBase: zscore
using Impute

"""
    filter_jongro(df)
Filter DataFrame by jongro station code (111123)    
"""
function filter_jongro(df)
    jongro_stn = 111123
    jongro_df = @from i in df begin
        @where i.stationCode == jongro_stn
        @select i
        @collect DataFrame
    end

    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid]
    for col in cols
        jongro_df[col] = Missings.coalesce.(jongro_df[col], 0.0)
    end

    return jongro_df
end

function get_nearjongro(df)
    stn_list = []
end

function save_jongro_df(input_path = "/input/input.csv")
    df = CSV.read(input_path)
    sort!(df, (:date, :stationCode))
    @show first(df, 5)
    j_df = filter_jongro(df)

    CSV.write("/input/jongro_single.csv", j_df)
end

function read_jongro(input_path="/input/jongro_single.csv")
    if Base.Filesystem.isfile(input_path) == false
        save_jongro_df()
    end

    df = CSV.read(input_path)
    
    @info "Start preprocessing..."
    flush(stdout); flush(stderr)
    df[:date] = ZonedDateTime.(df[:date], Dates.DateFormat("yyyy-mm-dd HH:MM:SSz"))

    # no and staitonCode must not have missing value
    @assert size(collect(skipmissing(df[:no])), 1) == size(df, 1)
    @assert size(collect(skipmissing(df[:stationCode])), 1) == size(df, 1)

    dropmissing!(df, [:no, :stationCode])
    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    airkorea_cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25]
    weather_cols = [:temp, :u, :v, :pres, :humid]

    @info "Imputing data..."
    flush(stdout); flush(stderr)
    allowmissing!(df, cols)
    for col in [:prep, :snow]
        df[col] = Missings.coalesce.(df[col], 0.0)
    end

    for col in airkorea_cols
        replace!(df[col], -999 => missing)
    end

    # convert -999 to missing
    for col in vcat(airkorea_cols, weather_cols)
        impute!(df[col], :interp)
    end

    # check missing values len_init_df
    for col in names(df)
        @assert size(df, 1) == size(collect(skipmissing(df[col])), 1)
    end
    dropmissing!(df, cols, disallowmissing=true)

    @show first(df, 5)
    flush(stdout); flush(stderr)

    df
end

function perm_df(df, permed_idx, col, labels)
    X = df[permed_idx, labels]
    Y = df[permed_idx, col]

    X, Y
end
