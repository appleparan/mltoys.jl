using DataFrames
using StatsBase

function standardize!(df::DataFrame, col::Symbol, new_col::Symbol)
    to_be_normalized = df[col]
    df[new_col] = zcore(to_be_normalized)
end 

standardize!(df::DataFrame, col::String, new_col::String) = standardize!(df, Symbol(col), Symbol(new_col))
standardize!(df::DataFrame, col::Symbol) = standardize!(df, col, col)
standardize!(df::DataFrame, col::String) = standardize!(df, Symbol(col))
