using DataFrames
using StatsBase: mean, std, zscore

# TODO : How to pass optional argument? 
"""
    standardize!(df, col, new_col)
Apply zscore (normalization) to dataframe `df`
"""
function standardize!(df::DataFrame, col::Symbol, new_col::Symbol) 
    to_be_normalized = df[col]
    df[new_col] = zscore(to_be_normalized)
end

function standardize!(df::DataFrame, col::Symbol, new_col::Symbol, μ::Real, σ::Real)
    to_be_normalized = df[col]
    df[new_col] = zscore(to_be_normalized, μ, σ)
end

standardize!(df::DataFrame, col::String, new_col::String) = standardize!(df, Symbol(eval(col)), Symbol(eval(new_col)))
standardize!(df::DataFrame, col::Symbol) = standardize!(df, col, col)
standardize!(df::DataFrame, col::String) = standardize!(df, Symbol(col))

"""
    exclude_elem(cols, target_col)
exclude element and return new splited array
"""
function exclude_elem(cols, target_col)
    new_cols = copy(cols)
    deleteat!(new_cols, new_cols .== target_col)

    new_cols
end