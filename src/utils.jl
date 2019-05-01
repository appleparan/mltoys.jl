using Base.Iterators: partition, zip
using Random

using CuArrays
using Dates, TimeZones
using DataFrames, Query
using Flux
using JuliaDB
using MicroLogging
using StatsBase: mean, std, zscore, mean_and_std

function mean_and_std_cols(df::DataFrame, cols::Array{Symbol, 1})
    syms = []
    types = []
    vals = []
    for col in cols
        μ, σ = mean_and_std(df[col])

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
    to_be_normalized = df[col]
    df[new_col] = hampel(to_be_normalized)
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

hampel!(df::DataFrame, col::Symbol, new_col::String) = hampel!(df, col, Symbol(eval(new_col)))
hampel!(df::DataFrame, col::String, new_col::String) = hampel!(df, Symbol(eval(col)), Symbol(eval(new_col)))
hampel!(df::DataFrame, col::Symbol) = hampel!(df, col, col)
hampel!(df::DataFrame, col::String) = hampel!(df, Symbol(col))

hampel!(df::DataFrame, col::Symbol, new_col::String, μ::Real, σ::Real) = hampel!(df, col, Symbol(eval(new_col)), μ, σ)
hampel!(df::DataFrame, col::String, new_col::String, μ::Real, σ::Real) = hampel!(df, Symbol(eval(col)), Symbol(eval(new_col)), μ, σ)
hampel!(df::DataFrame, col::Symbol, μ::Real, σ::Real) = hampel!(df, col, col, μ, σ)
hampel!(df::DataFrame, col::String, μ::Real, σ::Real) = hampel!(df, Symbol(col), μ, σ)

# TODO : How to pass optional argument? 
"""
    zscore!(df, col, new_col)
Apply zscore (normalization) to dataframe `df`
TODO: how to make μ and σ optional?
"""
function zscore!(df::DataFrame, col::Symbol, new_col::Symbol)
    to_be_normalized = df[col]
    df[new_col] = zscore(to_be_normalized)
end

function zscore!(df::DataFrame, col::Symbol, new_col::Symbol, μ::Real, σ::Real)
    to_be_normalized = df[col]
    df[new_col] = zscore(to_be_normalized, μ, σ)
end

zscore!(df::DataFrame, col::Symbol, new_col::String) = zscore!(df, col, Symbol(eval(new_col)))
zscore!(df::DataFrame, col::String, new_col::String) = zscore!(df, Symbol(eval(col)), Symbol(eval(new_col)))
zscore!(df::DataFrame, col::Symbol) = zscore!(df, col, col)
zscore!(df::DataFrame, col::String) = zscore!(df, Symbol(col))

zscore!(df::DataFrame, col::Symbol, new_col::String, μ::Real, σ::Real) = zscore!(df, col, Symbol(eval(new_col)), μ, σ)
zscore!(df::DataFrame, col::String, new_col::String, μ::Real, σ::Real) = zscore!(df, Symbol(eval(col)), Symbol(eval(new_col)), μ, σ)
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
exclude element and return new splited array
"""
function exclude_elem(cols, target_col)
    new_cols = copy(cols)
    deleteat!(new_cols, new_cols .== target_col)

    new_cols
end

"""
    split_df(df, sample_size)
split to `sample_size`d DataFrame 
"""
function split_df(df::DataFrame, sample_size::Integer)
    idxs = partition(1:size(df, 1), sample_size)
    # create array filled with undef and its size is length(idxs)
    
    idxs
end

"""
    window_df(df, sample_size)
create list of overlapped windowe index range 
"""
function window_df(df::DataFrame, sample_size::Integer, offset::Integer = 0)
    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    start_idxs = collect(1:(size(df, 1) - sample_size + 1))
    final_idxs = start_idxs .+ (sample_size - 1)
    idxs = []
    for (si, fi) in zip(start_idxs, final_idxs)
        push!(idxs, (si+offset):(fi+offset))
    end
    
    idxs
end

"""
    window_df(df, sample_size, start_date, end_date)
create list of overlapped windowe index range within date range
"""
function window_df(df::DataFrame, sample_size::Integer, _start_date::ZonedDateTime, _end_date::ZonedDateTime)
    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    # moreover,I should to round time to 1 hour unit
    start_date = ceil(max(_start_date, df.date[1]), Dates.Hour(1))
    end_date = floor(_end_date, Dates.Hour(1))
    if start_date > end_date
        throw(ArgumentError("invalid date range: $start_date ~ $end_date"))
    end

    # .value : to get `Dates` value 
    if Dates.Hour(end_date - start_date).value < sample_size
        throw(BoundsError("sample size($sample_size) is smaller than date range: $start_date ~ $end_date"))
    end
    
    new_df = df[(df.date .>= start_date) .& (df.date .<= end_date), :]
    offset = Dates.Hour(start_date - df.date[1]).value

    window_df(new_df, sample_size, offset)
end

"""
    window_df(df, sample_size, end_date)
create list of overlapped windowe index range within date range starts with 1970. 1. 1.
"""
function window_df(df::DataFrame, sample_size::Integer, end_date::ZonedDateTime)
    start_date = ZonedDateTime(1970, 1, 1, tz"Asia/Seoul")

    window_df(df, sample_size, start_date, end_date)
end

"""
    split_sizes3(total_size, batch_size)
"""
function split_sizes3(total_size::Integer, batch_size::Integer)
    # (train + valid) : test = 0.8 : 0.2
    # train : valid = 0.8 : 0.2
    # train : valid : test = 0.64 : 0.16 : 0.20
    train_size = round(total_size * 0.64)
    valid_size = round(total_size * 0.16)
    test_size = total_size - (train_size + valid_size)

    # at least contains single batch
    @assert train_size >= batch_size

    Int(train_size), Int(valid_size), Int(test_size)
end

"""
    split_sizes2(total_size, batch_size)
"""
function split_sizes2(total_size::Integer, batch_size::Integer)
    # (train + valid) : test = 0.8 : 0.2
    # train : valid = 0.8 : 0.2
    # train : valid : test = 0.64 : 0.16 : 0.20
    train_size = round(total_size * 0.8)
    valid_size = round(total_size * 0.2)

    # at least contains single batch
    @assert train_size >= batch_size

    Int(train_size), Int(valid_size)
end

create_chunk(xs, n) = collect(Iterators.partition(xs, n))
"""
    create_chunks(tot_size, train_size, valid_size, test_size, batch_size)
create chunks by batch_size that indicates index of sg_idxs or wd_idxs

expected sample result of chunks
    [[1, 2]
    [3, 4],
    [5]]
"""
function create_chunks(total_idx::Array{I, 1},
    train_size::Integer, valid_size::Integer, test_size::Integer, batch_size::Integer) where I<:Integer

    train_chnks = create_chunk(total_idx[1: train_size], batch_size)
    valid_chnks = create_chunk(total_idx[train_size + 1: train_size + valid_size], batch_size)
    test_chnks = create_chunk(total_idx[train_size + valid_size + 1: train_size + valid_size + test_size], batch_size)

    train_chnks, valid_chnks, test_chnks
end

"""
    create_chunks(tot_size, train_size, valid_size, batch_eof sg_idxs or wd_idxs

expected sample result of chunks
    [[1, 2]
    [3]]
"""
function create_chunks(total_idx::Array{I, 1},
    train_size::Integer, valid_size::Integer, batch_size::Integer) where I<:Integer

    train_chnks = create_chunk(total_idx[1: train_size], batch_size)
    valid_chnks = create_chunk(total_idx[train_size + 1: train_size + valid_size], batch_size)

    train_chnks, valid_chnks
end

"""
    create_idxs(tot_idx, train_size, valid_size, test_size)
create indexes that indicates index of sg_idxs or wd_idxs

expected sample result of idxs
    [1, 2, 3, 4, 5], [6, 7], [8, 9, 10]
"""
function create_idxs(tot_idx::Array{I, 1}, train_size::Integer, valid_size::Integer, test_size::Integer) where I<:Integer
    train_idxs = tot_idx[1: train_size]
    valid_idxs = tot_idx[train_size + 1: train_size + valid_size]
    test_idxs = tot_idx[train_size + valid_size + 1: train_size + valid_size + test_size]

    train_idxs, valid_idxs, test_idxs
end

"""
    create_idxs(tot_idx, train_size, valid_size)
create indexes that indicates index of sg_idxs or wd_idxs

expected sample result of idxs
    [1, 2, 3, 4, 5], [6, 7]
"""
function create_idxs(tot_idx::Array{I, 1}, train_size::Integer, valid_size::Integer) where I<:Integer
    train_idxs = tot_idx[1: train_size]
    valid_idxs = tot_idx[train_size + 1: train_size + valid_size]

    train_idxs, valid_idxs
end

"""
    getHoursLater(df, hours, last_date_str, date_fmt)
get `hours` rows of DataFrame `df` after given `date`, String will be automatically  conveted to ZonedDateTime Object
"""
function getHoursLater(df::DataFrame, hours::Integer, last_date_str::String, date_fmt::Dates.DateFormat=Dates.DateFormat("yyyy-mm-dd HH:MM:SSz"))
    last_date = ZonedDateTime(last_date_str, date_fmt)
    
    return getHoursLater(df, hours, last_date)
end

function getHoursLater(df::DataFrame, hours::Integer, last_date::ZonedDateTime)
    start_date = last_date

    # collect within daterange
    df_hours = df |> @filter(start_date < _.date <= (start_date + Hour(hours))) |> DataFrame
    
    df_hours
end

"""
    getX(df::DataFrame, idxs, features)
get X in Dataframe and construct X by flattening
"""
function getX(df::DataFrame, idxs::Array{I, 1}, features::Array{Symbol, 1}) where I<:Integer
    X = convert(Matrix, df[collect(idxs), features])
    
    # serialize (2D -> 1D)
    return vec(X)
end

getX(df::DataFrame, idxs, features::Array{String,1}) = getX(df, idxs, Symbol.(eval.(features)))

"""
    getY(X::DateFrame, hours)
get last date of X and construct Y with `hours` range
"""
function getY(df::DataFrame, idx::Array{I, 1},
    ycol::Symbol, output_size::Integer=24) where I<:Integer
    df_X = df[idx, :]
    last_date_of_X = df_X[end, :date]
    
    Y = getHoursLater(df, output_size, last_date_of_X)

    Y[:, ycol]
end

"""
    make_pairs(df, ycol, idx, features, hours)
create pairs in `df` along with `idx` (row) and `features` (columns)
output determined by ycol

"""
# Bundle images together with labels and group into minibatchess
function make_pairs(df::DataFrame, ycol::Symbol,
    idx::Array{I, 1}, features::Array{Symbol, 1},
    input_size::Integer, output_size::Integer) where I<:Integer
    X = getX(df, idx, features) |> gpu
    Y = getY(df, idx, ycol, output_size) |> gpu

    @assert length(X) == input_size
    @assert length(Y) == output_size
    @assert ndims(X) == 1
    @assert ndims(Y) == 1

    return (X, Y)
end

"""
    make_minibatch(input_pairs, batch_size, train_idx, valid_idx, test_idx)
make batch by sampling. size of batch (input_size, batch_size) 

[(input_single, output_single)...] =>
[   
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // single batch
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // single batch
    ((input_single, input_single, ..., input_single,),  (output_single, output_single, ...output_single,)), // single batch
]

do this for train_set, valid_set, test_set
each single batch is column stacked

"""
function make_minibatch(input_pairs::Array{T},
    chnks::Array{I, 1}, batch_size::Integer) where I<:Integer where T<:Tuple
    # input_pairs Array{Tuple{Array{Int64,1},Array{Int64,1}},1}
    X = []
    Y = []

    X = [pair[1] for pair in input_pairs[chnks]]
    Y = [pair[2] for pair in input_pairs[chnks]]
    X_size = length(X[1])
    Y_size = length(Y[1])
    # append zeros if chnks size is less than batch_size
    chnks_size = length(chnks)
    if chnks_size < batch_size
        for i in chnks_size+1:batch_size
            append!(X, [zeros(X_size)])
            append!(Y, [zeros(Y_size)])
        end
    end

    # (input_size * batch_size) x length(pairs), (output_size) x length(pairs)
    # Flux.batchseq : pad zero when size is lower than batch_size
    (Flux.batch(X), Flux.batch(Y))
end