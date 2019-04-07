using CSV
using DataFrames, Query, Missings
using Dates, TimeZones
using Random
using StatsBase: zscore

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
    
    #df = dropmissing(df, [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid], disallowmissing=true)
    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid]
    for col in cols
        df[col] = Missings.coalesce.(df[col], 0.0)
    end

    df[:date] = ZonedDateTime.(df[:date], Dates.DateFormat("yyyy-mm-dd HH:MM:SSz"))

    @show first(df, 5)

    df
end

function perm_df(df, permed_idx, col, labels)
    X = df[permed_idx, labels]
    Y = df[permed_idx, col]

    X, Y
end
