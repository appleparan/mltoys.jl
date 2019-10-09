"""
    mean_and_std_cols(df, cols)

find mean and std value in df[:, col]
"""
function mean_and_std_cols(df::DataFrame, cols::Array{Symbol, 1})
    
    syms = []
    types = []
    vals = []
    for col in cols
        μ, σ = mean_and_std(df[!, col])
        _min = minimum(df[!, col])
        _max = maximum(df[!, col])

        push!(syms, String(col))
        push!(types, "μ")
        push!(vals, μ)

        push!(syms, String(col))
        push!(types, "σ")
        push!(vals, σ)

        push!(syms, String(col))
        push!(types, "minimum")
        push!(vals, _min)

        push!(syms, String(col))
        push!(types, "maximum")
        push!(vals, _max)
    end

    μσ = ndsparse((
        symbol = syms,
        types = types),
        (value = vals,))

    μσ
end

"""
    hampel!([Z], X, μ, σ)

Compute the Hampel estimator of an array `X` with mean `μ` and standard deviation `σ`.
Hampel estimator formula follows as ``1/2*{tanh(0.01*((x - μ) / σ))+1.0}``.
If a destination array `Z` is provided, the scores are stored
in `Z` and it must have the same shape as `X`. Otherwise `X` is overwritten.

Detailed implementation structure is from `zscore` in `StatsBase` package
"""
function _hampel!(Z::AbstractArray, X::AbstractArray, μ::Real, σ::Real)
    
    # Z and X are assumed to have the same size
    iσ = inv(σ)

    for i = 1 : length(X)
        @inbounds Z[i] = (tanh(0.01 * (X[i] - μ) * iσ) + 1.0) * 0.5
    end

    return Z
end

@generated function _hampel!(Z::AbstractArray{S,N}, X::AbstractArray{T,N},
                             μ::AbstractArray, σ::AbstractArray) where {S,T,N}
    quote
        # Z and X are assumed to have the same size
        # μ and σ are assumed to have the same size, that is compatible with size(X)
        siz1 = size(X, 1)
        @nextract $N ud d->size(μ, d)
        if size(μ, 1) == 1 && siz1 > 1
            @nloops $N i d->(d>1 ? (1:size(X,d)) : (1:1)) d->(j_d = ud_d ==1 ? 1 : i_d) begin
                v = (@nref $N μ j)
                c = inv(@nref $N σ j)
                for i_1 = 1:siz1
                    # (@nref $N Z i) = ((@nref $N X i) - v) * c
                    (@nref $N Z i) = (tanh(0.01 * ((@nref $N X i) - v) * c) + 1.0) * 0.5
                end
            end
        else
            @nloops $N i X d->(j_d = ud_d ==1 ? 1 : i_d) begin
                # (@nref $N Z i) = ((@nref $N X i) - (@nref $N μ j)) / (@nref $N σ j)
                (@nref $N Z i) = (tanh(0.01 * ((@nref $N X i) - (@nref $N μ j)) / (@nref $N σ j)) + 1.0) * 0.5
            end
        end
        return Z
    end
end

function _hampel_chksize(X::AbstractArray, μ::AbstractArray, σ::AbstractArray)
    size(μ) == size(σ) || throw(DimensionMismatch("μ and σ should have the same size."))
    for i=1:ndims(X)
        dμ_i = size(μ,i)
        (dμ_i == 1 || dμ_i == size(X,i)) || throw(DimensionMismatch("X and μ have incompatible sizes."))
    end
end

function hampel!(Z::AbstractArray{ZT}, X::AbstractArray{T}, μ::Real, σ::Real) where {ZT<:AbstractFloat,T<:Real}
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _hampel!(Z, X, μ, σ)
end

function hampel!(Z::AbstractArray{<:AbstractFloat}, X::AbstractArray{<:Real},
                 μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real})
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _hampel_chksize(X, μ, σ)
    _hampel!(Z, X, μ, σ)
end

hampel!(X::AbstractArray{<:AbstractFloat}, μ::Real, σ::Real) = _hampel!(X, X, μ, σ)

function hampel(X::AbstractArray{T}, μ::Real, σ::Real) where T<:Real
    HT = typeof((tanh(0.01 * (zero(T) - zero(μ)) / one(σ)) + 1.0) * 0.5)
    _hampel!(Array{HT}(undef, size(X)), X, μ, σ)
end

function hampel(X::AbstractArray{T}, μ::AbstractArray{U}, σ::AbstractArray{S}) where {T<:Real,U<:Real,S<:Real}
    _hampel_chksize(X, μ, σ)
    HT = typeof((tanh(0.01 * (zero(T) - zero(μ)) / one(σ)) + 1.0) * 0.5)
    _hampel!(Array{HT}(undef, size(X)), X, μ, σ)
end

hampel(X::AbstractArray{<:Real}) = ((μ, σ) = mean_and_std(X); hampel(X, μ, σ))
hampel(X::AbstractArray{<:Real}, dim::Int) = ((μ, σ) = mean_and_std(X, dim); hampel(X, μ, σ))

"""
    hampel!(df, col, new_col)
Apply hampel estimators to dataframe `df`
"""
function hampel!(df::DataFrame, col::Symbol, new_col::Symbol)
    to_be_normalized = df[!, col]
    df[!, new_col] = hampel(to_be_normalized)
end

function hampel!(df::DataFrame, cols::Array{String, 1}, new_cols::Array{String, 1})
    for (col, new_col) in zip(cols, new_cols)
        hampel!(df, col, new_col)
    end
end

function hampel!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1})
    for (col, new_col) in zip(cols, new_cols)
        hampel!(df, col, new_col)
    end
end

hampel!(df::DataFrame, col::Symbol, new_col::String) = hampel!(df, col, Symbol(new_col))
hampel!(df::DataFrame, col::String, new_col::String) = hampel!(df, Symbol(col), Symbol(new_col))
hampel!(df::DataFrame, col::Symbol) = hampel!(df, col, col)
hampel!(df::DataFrame, col::String) = hampel!(df, Symbol(col))

hampel!(df::DataFrame, col::Symbol, new_col::String, μ::Real, σ::Real) = hampel!(df, col, Symbol(new_col), μ, σ)
hampel!(df::DataFrame, col::String, new_col::String, μ::Real, σ::Real) = hampel!(df, Symbol(col), Symbol(new_col), μ, σ)
hampel!(df::DataFrame, col::Symbol, μ::Real, σ::Real) = hampel!(df, col, col, μ, σ)
hampel!(df::DataFrame, col::String, μ::Real, σ::Real) = hampel!(df, Symbol(col), μ, σ)

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

zscore!(df::DataFrame, col::Symbol, new_col::String) = zscore!(df, col, Symbol(new_col))
zscore!(df::DataFrame, col::String, new_col::String) = zscore!(df, Symbol(col), Symbol(new_col))
zscore!(df::DataFrame, col::Symbol) = zscore!(df, col, col)
zscore!(df::DataFrame, col::String) = zscore!(df, Symbol(col))

zscore!(df::DataFrame, col::Symbol, new_col::String, μ::Real, σ::Real) = zscore!(df, col, Symbol(new_col), μ, σ)
zscore!(df::DataFrame, col::String, new_col::String, μ::Real, σ::Real) = zscore!(df, Symbol(col), Symbol(new_col), μ, σ)
zscore!(df::DataFrame, col::Symbol, μ::Real, σ::Real) = zscore!(df, col, col, μ, σ)
zscore!(df::DataFrame, col::String, μ::Real, σ::Real) = zscore!(df, Symbol(col), μ, σ)

function zscore!(df::DataFrame, cols::Array{String, 1}, new_cols::Array{String, 1})
    for (col, new_col) in zip(cols, new_cols)
        zscore!(df, col, new_col)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1})
    for (col, new_col) in zip(cols, new_cols)
        zscore!(df, col, new_col)
    end
end

function zscore!(df::DataFrame, cols::Array{String, 1}, new_cols::Array{String, 1}, μs::Array{Real, 1}, σs::Array{Real, 1})
    for (col, new_col, μ, σ) in zip(cols, new_cols, μs, σs)
        zscore!(df, col, new_col, μ, σ)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1}, μs::Array{Real, 1}, σs::Array{Real, 1})
    for (col, new_col, μ, σ) in zip(cols, new_cols, μs, σs)
        zscore!(df, col, new_col, μ, σ)
    end
end

"""
    min_max_scaling_cols(df, cols)

find mean and std value in df[:, col]
"""
function min_max_scaling!(df::DataFrame, cols::Array{String, 1}, new_cols::Array{String, 1},
    new_min::F = -1.0, new_max::F = 1.0) where F<:AbstractFloat
    min_max_scaling!(df, Symbol.(cols), Symbol.(new_cols), new_min, new_max)
end

function min_max_scaling!(df::DataFrame, cols::Array{Symbol, 1}, new_cols::Array{Symbol, 1},
    new_min::F = -1.0, new_max::F = 1.0) where F<:AbstractFloat

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
