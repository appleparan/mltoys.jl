"""
    extract_col_feats(df, cols)

find mean, std, minimum, maximum in df[:, col]
"""
function extract_col_statvals(df::DataFrame, cols::Array{Symbol, 1})
    syms = []
    types = []
    vals = []
    for col in cols
        μ, σ = mean_and_std(df[!, col])
        minval = minimum(df[!, col])
        maxval = maximum(df[!, col])

        push!(syms, String(col))
        push!(types, "μ")
        push!(vals, μ)

        push!(syms, String(col))
        push!(types, "σ")
        push!(vals, σ)

        push!(syms, String(col))
        push!(types, "minimum")
        push!(vals, minval)

        push!(syms, String(col))
        push!(types, "maximum")
        push!(vals, maxval)
    end

    ndsparse((
        symbol = syms,
        types = types),
        (value = vals,))
end

# TODO : How to pass optional argument to nested function?
"""
    zscore!(df, col, new_col)

Apply zscore (normalization) to dataframe `df`
TODO: how to make μ and σ optional?
"""
function zscore!(df::DataFrame, col::Symbol, new_col::Symbol)
    to_be_normalized = df[!, col]
    df[!, new_col] = zscore(to_be_normalized)
end

function zscore!(df::DataFrame, col::Symbol, new_col::Symbol, μ::Real, σ::Real)
    to_be_normalized = df[!, col]
    df[!, new_col] = zscore(to_be_normalized, μ, σ)
end

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1})
    for (col, new_col) in zip(cols, new_cols)
        zscore!(df, col, new_col)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1}, μs::Array{Real, 1}, σs::Array{Real, 1})
    for (col, new_col, μ, σ) in zip(cols, new_cols, μs, σs)
        zscore!(df, col, new_col, μ, σ)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1}, μσs::Array{Real, 1})
    for (col, new_col) in zip(cols, new_cols)
        μ, σ = μσs[String(ycol), "μ"].value, μσs[String(ycol), "σ"].value
        zscore!(df, col, new_col, μ, σ)
    end
end

"""
    min_max_scaling_cols(df, cols)

find mean and std value in df[:, col]
"""
function min_max_scaling!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1},
    new_min::F, new_max::F) where F<:AbstractFloat

    for (col, new_col) in zip(cols, new_cols)
        min_max_scaling!(df, col, new_col, new_min, new_max)
    end
end

function min_max_scaling!(df::DataFrame, col::Symbol, new_col::Symbol,
    new_min::F, new_max::F) where F<:AbstractFloat

    to_be_normalized = df[!, col]
    df[!, new_col] = new_min .+ (new_max - new_min) .*
        (to_be_normalized .- minimum(to_be_normalized)) ./
        (maximum(to_be_normalized) .- minimum(to_be_normalized))
end

"""
    exclude_elem(cols, target_col)

Exclude element and return new splited array
"""
function exclude_elem(cols, target_col)
    new_cols = copy(cols)
    deleteat!(new_cols, new_cols .== target_col)

    new_cols
end

"""
    findrow(df, col, val)

Find fist row number in df[:, `col`] as `val` by brute-force
"""
function findrow(df::DataFrame, col::Symbol, val::Union{<:Real, DateTime, ZonedDateTime})
    
    idx = 0
    for row in eachrow(df)
        idx += 1
        if (row[col] == val)
            return idx
        end
    end

    idx = 0

    idx
end

"""
    WHO_PM10(val::Real)
    return WHO PM10 level

1: Good
2: Normal
3: Bad
4: Very Bad
"""
function WHO_PM10(val::Real)
    if 0 <= val < 31
        return 1
    elseif 31 <= val < 81
        return 2
    elseif 81 <= val < 151
        return 3
    else
        return 4
    end
end

"""
    WHO_PM25(val::Real)
    return WHO PM25 level

1: Good
2: Normal
3: Bad
4: Very Bad
"""
function WHO_PM25(val::Real)
    if 0 <= val < 16
        return 1
    elseif 16 <= val < 36
        return 2
    elseif 36 <= val < 76
        return 3
    else
        return 4
    end
end
