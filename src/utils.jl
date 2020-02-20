"""
    extract_col_feats(df, cols)

find mean, std, minimum, maximum in df[!, col]
default value of columns are all numeric columns except date
"""
function extract_col_statvals(df::DataFrame, cols::Array{Symbol, 1})
    syms = []
    types = []
    vals = []
    for col in cols
        μ, σ = mean_and_std(skipmissing(df[!, col]))
        minval = minimum(skipmissing(df[!, col]))
        maxval = maximum(skipmissing(df[!, col]))

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


"""
    zscore!(df, col, new_col)

Apply zscore (normalization) to dataframe `df`
No need to implement `zscore`, just implement `zscore!``.
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

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1}, μσs::Array{Real, 1})
    for (col, new_col) in zip(cols, new_cols)
        μ, σ = μσs[String(col), "μ"].value, μσs[String(col), "σ"].value
        zscore!(df, col, new_col, μ, σ)
    end
end

"""
    unzscore(A, μ, σ)

unzscore in Array using given μ and σ
"""
unzscore(A, μ::Real, σ::Real) = A .* σ .+ μ

"""
    unzscore!(df, zcol, ocol, μ, σ)

revert zscore normalization (single column, zcol -> ocol)
"""
function unzscore!(df::DataFrame, zcol::Symbol, ocol::Symbol, μ::Real, σ::Real)
    df[!, ocol] = unzscore(df[!, zcol], μ, σ)
end

"""
    unzscore!(df, zcols, ocols, μσs)

revert zscore normalization (single column, zcol -> ocol)
"""
function unzscore!(df::DataFrame, zcols::Array{Symbol, 1}, ocols::Array{Symbol, 1}, stattb::AbstractNDSparse)
    for (zcol, ocol) in zip(zcols, ocols)
        unzscore!(df, zcol, ocol,
            stattb[String(ocol), "μ"].value, stattb[String(ocol), "σ"].value)
    end
end

minmax_scaling(X::AbstractVector, a::F, b::F) where F<:AbstractFloat =
    a .+ (b - a) .* (X .- minimum(X)) ./ (maximum(X) - minimum(X))

minmax_scaling!(df::DataFrame, ocol::Symbol, mcol::Symbol,
    a::F, b::F) where F<:AbstractFloat =
        df[!, mcol] = minmax_scaling(df[!, ocol], a, b)

"""
    minmax_scaling!(df, ocols, mcols, a, b)

min-max normalization from df[!, ocols] to df[!, mcols], new range is (a, b)
https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

Y = a + (X - min(X)) * (b - a) / (maximum(X) - minimum(X))
"""
function minmax_scaling!(df::DataFrame, ocols::Array{Symbol, 1}, mcols::Array{Symbol, 1},
    a::F, b::F) where F<:AbstractFloat

    for (ocol, mcol) in zip(ocols, mcols)
        minmax_scaling!(df, ocol, mcol, a, b)
    end
end

unminmax_scaling(Y::AbstractVector, minY::Real, maxY::Real, a::F, b::F) where F<:AbstractFloat = 
    (Y .- a) .* ((maxY - minY) / (b - a)) .+ minY

"""
    unminmax_scaling!(df, mcol, ocol,  minY, maxY, a, b)

revert min-max normalization (single column)
"""
function unminmax_scaling!(df::DataFrame,
    mcol::Symbol, ocol::Symbol, minY::Real, maxY::Real, a::F, b::F) where F<:AbstractFloat
    df[!, ocol] = unminmax_scaling(df[!, mcol], minY, maxY, a, b)
end

"""
    unminmax_scaling!(df, mcols, ocols, minY, maxY, a, b)

revert min-max normalization (multiple column)
"""
function unminmax_scaling!(df::DataFrame, zcols::Array{Symbol, 1}, ocols::Array{Symbol, 1},
    stattb::AbstractNDSparse, a::F, b::F) where F<:AbstractFloat
    for (zcol, ocol) in zip(zcols, ocols)
        unminmax_scaling!(df, zcol, ocol,
        stattb[String(ocol), "minimum"].value, stattb[String(ocol), "maximum"].value, a, b)
    end
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

Find fist row number in df[!, `col`] as `val` by brute-force
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
