using CSV
using DataFrames, Query
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

    dropmissing(jongro_df)
end

function get_nearjongro(df)
    stn_list = []
end

function save_jongro_df(input_path = "/input/input.csv")
    df = CSV.read(input_path)

    j_df = get_jongro(df)
    CSV.write("/input/jongro_single.csv", j_df)
end

function read_jongro(input_path="/input/jongro_single.csv")
    df = CSV.read(input_path)
    @show first(df, 5)
    @show size(df)
    df
end

"""
    standardize_df!(df, cols, prefix)
Standardize dataframe columns and save with new column name with given `prefix`
"""
function standardize_df!(df::DataFrame, cols::Array{String}, prefix::String="")
    for col in cols 
        new_col = prefix * col
        standardize!(df, col, new_col)
    end
end

function perm_df(df, permed_idx, col, labels)
    X = df[permed_idx, labels]
    Y = df[permed_idx, col]

    X, Y
end
