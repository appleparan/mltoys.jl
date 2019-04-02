using CSV
using DataFrames, Query
using Random
using StatsBase: zscore

function read_csv(input_path = "/home/appleparan/input/input.csv")
    df = CSV.read(input_path)

    return df
end

function get_jongro(df)
    jongro_stn = 111123
    jongro_df = @from i in df begin
        @where i.stationCode == jongro_stn
        @select i
        @collect DataFrame
    end

    jongro_df
end

function get_nearjongro(df)
    stn_list = []
end

function jongro_df()
    df = read_csv("/input/input.csv")
    j_df = get_jongro(df)
    CSV.write("/input/jongro_single.csv", j_df)
end


jongro_df()