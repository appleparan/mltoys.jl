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

        push!(syms, String(col))
        push!(types, "μ")
        push!(vals, μ)

        push!(syms, String(col))
        push!(types, "σ")
        push!(vals, σ)
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
    exclude_elem(cols, target_col)

Exclude element and return new splited array
"""
function exclude_elem(cols, target_col)
    new_cols = copy(cols)
    deleteat!(new_cols, new_cols .== target_col)

    new_cols
end

"""
    split_df(df, sample_size)

Split Dataframe df `sample_size`d  
"""
function split_df(df::DataFrame, sample_size::Integer)
    idxs = partition(1:size(df, 1), sample_size)
    # create array filled with undef and its size is length(idxs)
    
    idxs
end

"""
    window_df(df, sample_size)

Create list of overlapped windowe index range 
"""
function window_df(df::DataFrame, sample_size::I, output_size::I,
    offset::I = zero(I)) where I <: Integer

    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    start_idxs = collect(1:(size(df, 1) - sample_size - output_size + 1))
    final_idxs = start_idxs .+ (sample_size + output_size - 1)
    idxs = UnitRange{I}[]
    for (si, fi) in zip(start_idxs, final_idxs)
        push!(idxs, (si+offset):(fi+offset))
    end
    
    idxs
end

"""
    window_df(df, sample_size, start_date, end_date)

Create list of overlapped windowe index range within date range
"""
function window_df(df::DataFrame, sample_size::I, output_size::I,
    _start_date::ZonedDateTime, _end_date::ZonedDateTime)  where I <: Integer

    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    # moreover,I should to round time to 1 hour unit
    start_date = ceil(max(_start_date, df.date[1]), Dates.Hour(1))
    end_date = floor(min(_end_date, df.date[end]), Dates.Hour(1))
    if start_date > end_date
        throw(ArgumentError("invalid date range: $start_date ~ $end_date"))
    end

    # .value : to get `Dates` value 
    if Dates.Hour(end_date - start_date).value < sample_size + output_size
        throw(BoundsError("sample size($sample_size) is smaller than date range: $start_date ~ $end_date"))
    end

    new_df = df[(df.date .>= start_date) .& (df.date .<= end_date), :]
    # offset for subDataFrame
    offset = Dates.Hour(start_date - df.date[1]).value

    window_df(new_df, sample_size, output_size, offset)
end

"""
    window_df(df, sample_size, end_date)

Create list of overlapped window index range within date range starts with 1970. 1. 1.
"""
function window_df(df::DataFrame, sample_size::Integer, output_size::Integer, end_date::ZonedDateTime)
    start_date = ZonedDateTime(1970, 1, 1, tz"Asia/Seoul")

    window_df(df, sample_size, output_size, start_date, end_date)
end

"""
    split_sizes3(total_size, batch_size)

Split `total_size`for train/valid/test set with size checking compared to `batch_size`

# Examples
```julia-repl
julia> split_sizes3(100, 10)
(64, 16, 20)

julia> split_sizes3(10, 20)
ERROR: AssertionError: train_size >= batch_size
Stacktrace:
 [1] split_sizes3(::Int64, ::Int64) at ./REPL[85]:5
 [2] top-level scope at none:0
````
"""
function split_sizes3(total_size::I, batch_size::I) where I <: Integer
    train_size, valid_size, test_size = split_sizes3(total_size)

    # at least contains single batch
    @assert train_size >= batch_size

    train_size, valid_size, test_size
end

"""
    split_sizes3(total_size)

Split `total_size` for train/valid/test set (0.64:0.16:0.20)

# Default ratios
(train + valid) : test = 0.8 : 0.2
train : valid = 0.8 : 0.2
train : valid : test = 0.64 : 0.16 : 0.20

# Examples
```julia-repl
julia> split_sizes3(100)
(64, 16, 20)
```
"""
function split_sizes3(total_size::I,
    train_ratio::AbstractFloat = 0.64,
    valid_ratio::AbstractFloat = 0.16) where I <: Integer

    train_size = round(total_size * train_ratio)
    valid_size = round(total_size * valid_ratio)
    test_size = total_size - (train_size + valid_size)

    @assert train_size + valid_size + test_size == total_size

    I(train_size), I(valid_size), I(test_size)
end

"""
    split_sizes2(total_size, batch_size)

Split `total_size` for train/valid set with size checking compared to `batch_size`

# Default ratios
train : valid = 0.8 : 0.2

# Examples
```julia-repl
julia> split_sizes2(100)
(80, 20)

julia> split_sizes2(10, 20)
ERROR: AssertionError: train_size >= batch_size
Stacktrace:
 [1] split_sizes2(::Int64, ::Int64) at ./REPL[85]:5
 [2] top-level scope at none:0
```
"""
function split_sizes2(total_size::I, batch_size::I) where I <: Integer

    train_size, valid_size = split_sizes2(total_size)

    # at least contains single batch
    @assert train_size >= batch_size

    train_size, valid_size
end

"""
    split_sizes2(total_size)

Split `total_size` for train/valid set (0.8:0.2)

# Default ratios
train : valid = 0.8 : 0.2

# Examples
```julia-repl
julia> split_sizes2(100)
(80, 20)
```
"""
function split_sizes2(total_size::I,
    train_ratio::AbstractFloat = 0.8) where I <: Integer

    train_size = round(total_size * train_ratio)
    valid_size = total_size - train_size

    @assert train_size + valid_size == total_size

    I(train_size), I(valid_size)
end

"""
    create_idxs(total_idxs, train_size, valid_size, test_size)

Split `total_idxs` to train/valid/test set to indicate indices of window

# Examples
```julia-repl
julia> total_size = 10; window_size = 12; total_idxs = [i:(i+window_size-1) for i in 1:total_size]
10-element Array{UnitRange{Int64},1}:
 1:12
 2:13
 3:14
 4:15
 5:16
 6:17
 7:18
 8:19
 9:20
 10:21

julia> create_idxs(total_idxs, 6, 2, 2)
(UnitRange{Int64}[1:12, 2:13, 3:14, 4:15, 5:16, 6:17], UnitRange{Int64}[7:18, 8:19], UnitRange{Int64}[9:20, 10:21])
```
"""
function create_idxs(total_idxs::Array{<:UnitRange{I}, 1},
    train_size::I, valid_size::I, test_size::I) where I <: Integer

    train_idxs = total_idxs[1:train_size]
    valid_idxs = total_idxs[(train_size + 1):(train_size + valid_size)]
    test_idxs = total_idxs[(train_size + valid_size + 1):(train_size + valid_size + test_size)]

    train_idxs, valid_idxs, test_idxs
end

"""
    create_idxs(total_idxs, train_size, valid_size)

Split `total_idxs` to train/valid set to indicate indices of window

# Examples
```julia-repl
julia> total_size = 10; window_size = 12; total_idxs = [i:(i+window_size-1) for i in 1:total_size]
10-element Array{UnitRange{Int64},1}:
 1:12
 2:13
 3:14
 4:15
 5:16
 6:17
 7:18
 8:19
 9:20
 10:21

julia> create_idxs(total_idxs, 8, 2)
(UnitRange{Int64}[1:12, 2:13, 3:14, 4:15, 5:16, 6:17, 7:18, 8:19], UnitRange{Int64}[9:20, 10:21])
```
"""
function create_idxs(total_idxs::Array{<:UnitRange{I}, 1},
    train_size::I, valid_size::I) where I <: Integer

    train_idxs = total_idxs[1:train_size]
    valid_idxs = total_idxs[(train_size + 1):(train_size + valid_size)]

    train_idxs, valid_idxs
end

"""
    create_idxs(total_idxs, test_size)

`total_idxs` to test set to indicate indices of window

# Examples
```julia-repl
julia> total_size = 10; window_size = 12; total_idxs = [i:(i+window_size-1) for i in 1:total_size]
10-element Array{UnitRange{Int64},1}:
 1:12
 2:13
 3:14
 4:15
 5:16
 6:17
 7:18
 8:19
 9:20
 10:21

julia> create_idxs(total_idxs, 2)
2-element Array{UnitRange{Int64},1}:
 1:12
 2:13
```
"""
function create_idxs(total_idxs::Array{<:UnitRange{I}, 1},
    test_size::I) where I <: Integer

    test_idxs = total_idxs[1:test_size]

    test_idxs
end

"""
    create_chunks(total_idxs, train_size, valid_size, test_size, batch_size)

Create chunks which size is `batch_size` of train/valid/test set using in minibatch

See also: [`split_sizes3`](@ref)

# Example
```julia-repl
julia> total_size = 10; window_size = 4; batch_size = 3; total_idxs = [i:(i+window_size-1) for i in 1:total_size]
10-element Array{UnitRange{Int64},1}:
 1:4
 2:5
 3:6
 4:7
 5:8
 6:9
 7:10
 8:11
 9:12
 10:13

julia> train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
(6, 2, 2)

julia> create_chunks(total_idxs, train_size, valid_size, test_size, batch_size)
(Array{UnitRange{Int64},1}[[1:4, 2:5, 3:6], [4:7, 5:8, 6:9]], Array{UnitRange{Int64},1}[[7:10, 8:11]], Array{UnitRange{Int64},1}[[9:12, 10:13]])
```
"""
function create_chunks(total_idxs::Array{<:UnitRange{I}, 1},
    train_size::I, valid_size::I, test_size::I, batch_size::I) where I <: Integer

    train_chnks = create_chunk(total_idxs[1:train_size], batch_size)
    valid_chnks = create_chunk(total_idxs[(train_size + 1):(train_size + valid_size)], batch_size)
    test_chnks = create_chunk(total_idxs[(train_size + valid_size + 1):(train_size + valid_size + test_size)], batch_size)

    train_chnks, valid_chnks, test_chnks
end

"""
    create_chunks(total_idxs, train_size, valid_size, batch_size)

Create chunks which size is `batch_size` of train/valid set using in minibatch

See also: [`split_sizes2`](@ref)

# Example
```julia-repl
julia> total_size = 10; window_size = 4; batch_size = 3; total_idxs = [i:(i+window_size-1) for i in 1:total_size]
10-element Array{UnitRange{Int64},1}:
 1:4
 2:5
 3:6
 4:7
 5:8
 6:9
 7:10
 8:11
 9:12
 10:13

julia> train_size, valid_size = split_sizes2(total_size, batch_size)
(8, 2)

julia> create_chunks(total_idxs, train_size, valid_size, batch_size)
(Array{UnitRange{Int64},1}[[1:4, 2:5, 3:6], [4:7, 5:8, 6:9], [7:10, 8:11]], Array{UnitRange{Int64},1}[[9:12, 10:13]])
```
"""
function create_chunks(total_idx::Array{<:UnitRange{I}, 1},
    train_size::I, valid_size::I, batch_size::I) where I <: Integer

    train_chnks = create_chunk(total_idx[1:train_size], batch_size)
    valid_chnks = create_chunk(total_idx[(train_size + 1):(train_size + valid_size)], batch_size)

    train_chnks, valid_chnks
end

"""
    create_chunk(xs, n)

Simply split xs by n, basic function of create_chunks

See also: [`create_chunks`](@ref)
# Example
```julia-repl
julia> create_chunk(collect(1:10), 2)
5-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
 [9, 10]

julia> create_chunk(collect(1:10), 3)
4-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [4, 5, 6]
 [7, 8, 9]
 [10]
```
"""
create_chunk(xs, n) = collect(Iterators.partition(xs, n))

"""
    getHoursLater(df, hours, last_date_str, date_fmt)

Get `hours` rows of DataFrame `df` after given `last_date`.
Date String will be automatically conveted to ZonedDateTime Object
"""
function getHoursLater(df::DataFrame, hours::Integer, last_date_str::String, date_fmt::Dates.DateFormat=Dates.DateFormat("yyyy-mm-dd HH:MM:SSz"))
    last_date = ZonedDateTime(last_date_str, date_fmt)
    
    return getHoursLater(df, hours, last_date)
end

"""
    getHoursLater(df, hours, last_date)

Get `hours` rows of DataFrame `df` after given `last_date`
"""
function getHoursLater(df::DataFrame, hours::Integer, last_date::ZonedDateTime)
    start_date = last_date

    # collect within daterange
    if typeof(df[1, :date]) == DateTime
        start_date = DateTime(start_date)
    end
    
    df_hours = DataFramesMeta.@linq df |>
                    where(:date .> start_date, :date .<= (start_date + Hour(hours)))

    df_hours
end


"""
    remove_sparse_input!(X, Y, ratio = 0.5)

Remove this pair if sparse ratio is greather than given `ratio`.
"""
function remove_sparse_input!(X, Y, missing_ratio=0.5)
    @assert 0.0 <= missing_ratio <= 1.0

    invalid_idxs = UnitRange[]
    size_y = length(Y)
    size_m = length(findall(_y -> ismissing(_y) || _y == zero(_y), Y))
    
    if (size_m / size_y) >= missing_ratio
        # set X and Y to zeros
        X = fill!(X, zero(X[1, 1]))
        Y = fill!(Y, zero(Y[1, 1]))
    end

    nothing
end


"""
    getX_DNN(df::DataFrame, idx, features, sample_size)

Get 2D input X from DataFrame by idx
"""
function getX(df::DataFrame, idx::UnitRange{I}, features::Array{Symbol, 1}, sample_size::I) where I <: Integer
    X = convert(Matrix, df[idx, features])

    Matrix(X[1:sample_size, :])
end

"""
    getY(df, idx, ycol, sample_size, output_size)

Get last date of X and retrive Y after `last_date`
The size of Y should be output_size
"""
function getY(df::DataFrame, idx::UnitRange{I},
    ycol::Symbol, sample_size::I, output_size::I) where I <: Integer

    df_X = df[idx, :]
    last_date_of_X = df_X[sample_size, :date]

    Y = getHoursLater(df, output_size, last_date_of_X)

    Array(Y[:, ycol])
end

"""
    make_pair_DNN(df, ycol, idx, features, hours)

Create single pair in `df` along with `idx` (row) and `features` (columns)
output determined by ycol

input_size = sample size * num_selected_columns

"""
# Bundle images together with labels and group into minibatchess
function make_pair_DNN(df::DataFrame, ycol::Symbol,
    idx::UnitRange{I}, features::Array{Symbol, 1},
    sample_size::I, output_size::I) where I <: Integer
    _X = getX(df, idx, features, sample_size)
    _Y = getY(df, idx, ycol, sample_size, output_size)

    # 2D matrix to 1D array
    _X = vec(_X) |> gpu
    _Y = _Y |> gpu
    @assert length(_X) == sample_size * length(features)
    @assert length(_Y) == output_size
    @assert ndims(_X) == 1
    @assert ndims(_Y) == 1

    return (_X, _Y)
end

"""
    make_minibatch_DNN(df, ycol, chnks, features, sample_size, output_size, batch_size, Float32)

Create batch consists of pairs in `df` along with `idx` (row) and `features` (columns)
The batch is higher dimensional input consists of multiple pairs

single pairs to multiple 
pairs = [(X_1, Y_1), (X_2, Y_2), ...]

[(input_single, output_single)...] =>
[   
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // minibatch
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // minibatch
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // minibatch
]

and make pairs to be column stacked

# Example


batch_size = 5
pairs = [
            ([ 1, 2, 3, 4], [ 5, 6]),
            ([ 7, 8, 9,10], [11,12]),
            ([13,14,15,16], [17,18]),
            ([19,20,21,22], [23,24]),
            ([25,26,27,28], [29,30])]

become

([1,  7, 13, 19, 25;
  2,  8, 14, 20, 26;
  3,  9, 15, 21, 27;
  4, 10, 16, 22, 28], 
 [5, 11, 17, 23, 29;
  6, 12, 18, 24, 30])
"""
function make_batch_DNN(df::DataFrame, ycol::Symbol,
    chnks::Array{<:UnitRange{I}, 1}, features::Array{Symbol, 1},
    sample_size::I, output_size::I, batch_size::I,
    missing_ratio::AbstractFloat=0.5, T::DataType=Float32) where I <: Integer

    X = zeros(T, sample_size * length(features), batch_size)
    Y = zeros(T, output_size, batch_size)

    # input_pairs Array{Tuple{Array{Int64,1},Array{Int64,1}},1}
    for (i, idx) in enumerate(chnks)
        _X = vec(getX(df, idx, features, sample_size))
        _Y = getY(df, idx, ycol, sample_size, output_size)

        # fill zeros if data is too sparse (too many misisngs)
        # only in training
        remove_sparse_input!(_X, _Y)

        # make X & Y as column stacked batch
        X[:, i] = _X
        Y[:, i] = _Y
    end

    X = X |> gpu
    Y = Y |> gpu

    (X, Y)
end

"""
    is_sparse_Y(Y, ratio = 0.5)

Check Y is sparse, 0s are more than `raito`
"""
function is_sparse_Y(Y, missing_ratio=0.5)

    @assert 0.0 <= missing_ratio <= 1.0

    size_y = length(Y)
    size_m = length(findall(_y -> ismissing(_y) || _y == 0.0, Y))

    if (size_m / size_y) >= missing_ratio
        # if use deleteat here, it manipulates pairs while loop and return wrong results
        return true
    end
    
    false
end

"""
    make_input_LSTM(df, ycol, idx, features, hours)

Create pairs in `df` along with `idx` (row) and `features` (columns)
output determined by ycol
pairs doesn't need to consider batch_size

A size of LSTM X input must be (sample size, num_selected_columns)
A size of LSTM Y output must be (output_size,)

https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
"""
# Bundle images together with labels and group into minibatchess
# idxs would be < Array{Integer, 1} or < Array{UnitRange{Integer}, 1}
function make_input_LSTM(df::DataFrame, ycol::Symbol,
    idxs::Array{<:UnitRange{I}, 1}, features::Array{Symbol, 1},
    sample_size::I, output_size::I,
    p::Progress=Progress(length(idxs))) where I <: Integer

    X = Array{Real}(undef, 0, sample_size, length(features))
    Y = Array{Real}(undef, 0, output_size)

    for idx in idxs
        _X = getX(df, idx, features, sample_size)
        _Y = getY(df, idx, ycol, sample_size, output_size)

        @assert size(_X) == (sample_size, length(features))
        @assert size(_Y) == (output_size,)

        # even sparse, progressmeter should be go on
        ProgressMeter.next!(p)

        if is_sparse_Y(_Y)
            continue
        end

        X = cat(X, reshape(_X, 1, size(_X)[1], size(_X)[2]), dims=1)
        Y = cat(Y, reshape(_Y, 1, size(_Y)[1]), dims=1)
    end

    X = X |> gpu
    Y = Y |> gpu

    X, Y
end

"""
    findrow(df, col, val)

Find fist row number in df[:, `col`] as `val` by brute-force
"""
function findrow(df::DataFrame, col::Symbol, val::T) where T <: Real
    
    idx = zero(T)
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
