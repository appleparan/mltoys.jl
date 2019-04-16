using Base.Iterators: partition, zip

using CuArrays
using Dates, TimeZones
using DataFrames, Query
using Flux
using JuliaDB
using Random
using StatsBase: mean, std, zscore, mean_and_std

function mean_and_std_cols(df::DataFrame, cols::Array{Symbol})
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

function zscore!(df::DataFrame, cols::Array{String}, new_cols::Array{String})
    for (col, new_col) in zip(cols, new_cols)
        zscore!(df, col, new_col)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol}, new_cols::Array{Symbol})
    for (col, new_col) in zip(cols, new_cols)
        zscore!(df, col, new_col)
    end
end

function zscore!(df::DataFrame, cols::Array{String}, new_cols::Array{String}, μs::Array{Real}, σs::Array{Real})
    for (col, new_col, μ, σ) in zip(cols, new_cols, μs, σs)
        zscore!(df, col, new_col, μ, σ)
    end
end

function zscore!(df::DataFrame, cols::Array{Symbol}, new_cols::Array{Symbol}, μs::Array{Real}, σs::Array{Real})
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
function split_df(df::DataFrame, sample_size::Integer = 72)
    idxs = partition(1:size(df, 1), sample_size)
    # create array filled with undef and its size is length(idxs)
    
    idxs
end

"""
    window_df(df, sample_size)
create list of overlapped windowe index range 
"""
function window_df(df::DataFrame, sample_size::Integer = 72)
    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    start_idxs = collect(1:(size(df, 1) - 2 * sample_size))
    final_idxs = start_idxs .+ (sample_size - 1)
    idxs = []
    for (si, fi) in zip(start_idxs, final_idxs)
        push!(idxs, si:fi)
    end
    
    idxs
end

"""
    split_sizes(total_size, batch_size)
"""
function split_sizes(total_size::Integer, batch_size::Integer)
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

create_chunk(xs, n) = collect(Iterators.partition(xs, n))
"""
    create_chunks(tot_size, train_size, valid_size, test_size, batch_size)
create chunks by batch_size that indicates index of sg_idxs or wd_idxs

expected sample result of chunks
    [[1, 2]
    [3, 4],
    [5]]
"""
function create_chunks(total_idx::Array{Int64},
    train_size::Integer, valid_size::Integer, test_size::Integer, batch_size::Integer)

    train_chnks = create_chunk(total_idx[1: train_size], batch_size)
    valid_chnks = create_chunk(total_idx[train_size + 1: train_size + valid_size], batch_size)
    test_chnks = create_chunk(total_idx[train_size + valid_size + 1: end], batch_size)

    #=
    # if last element size is less than batch_size, drop it
    function drop_small_batch!(_idxs, _size)
        if length(_idxs[end]) < _size
            deleteat!(_idxs, length(_idxs))
        end
    end
    drop_small_batch!(train_chnks, batch_size)
    drop_small_batch!(valid_chnks, batch_size)
    drop_small_batch!(test_chnks, batch_size)
    =#
    train_chnks, valid_chnks, test_chnks
end

"""
    create_idxs(tot_idx, train_size, valid_size, test_size)
create indexes that indicates index of sg_idxs or wd_idxs

expected sample result of idxs
    [1, 2, 3, 4, 5]
"""
function create_idxs(tot_idx::Array{Int64}, train_size::Integer, valid_size::Integer, test_size::Integer)
    train_idxs = tot_idx[1: train_size]
    valid_idxs = tot_idx[train_size + 1: train_size + valid_size]
    test_idxs = tot_idx[train_size + valid_size + 1: end]

    train_idxs, valid_idxs, test_idxs
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
function getX(df::DataFrame, idxs, features::Array{Symbol})
    X = convert(Matrix, df[collect(idxs), features])
    
    # serialize (2D -> 1D)
    return vec(X)
end

getX(df::DataFrame, idxs, features::Array{String,1}) = getX(df, idxs, Symbol.(eval.(features)))

"""
    getY(X::DateFrame, hours)
get last date of X and construct Y with `hours` range
"""
function getY(df::DataFrame, idx::Array{T},
    ycol::Symbol, output_size::Integer=24) where T <: Integer
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
    idx::Array{T}, features::Array{Symbol},
    input_size::T, output_size::T) where T <: Integer
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
    chnks::Array{I,1}, batch_size::I) where I<:Integer where T<:Tuple
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