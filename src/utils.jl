using Base.Iterators: partition, zip

using DataFrames, Query
using StatsBase: mean, std, zscore
using Dates, TimeZones

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

"""
    split_df(df, sample_size)
split to `sample_size`d DataFrame 
"""
function split_df(df::DataFrame, sample_size::Integer = 72)
    idxs = partition(1:size(df, 1), sample_size)
    # create array filled with undef and its size is length(idxs)
    df_arr = Array{DataFrame}(undef, length(idxs))

    for (i, idx) in zip(1:length(idxs), idxs)
        df_arr[i] = df[idx, :]
    end
    
    df_arr, idxs
end

"""
    window_df(df, sample_size)
create overlapped windowed df
"""
function window_df(df::DataFrame, sample_size::Integer = 72)
    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    idxs = collect(1:(size(df, 1) - 2 * sample_size))
    # create array filled with undef and its size is length(idxs)
    df_arr = Array{DataFrame}(undef, length(idxs))

    for (i, idx) in zip(1:length(idxs), idxs)
        df_arr[i] = df[idx:(idx+sample_size-1), :]
    end
    
    df_arr, idxs
end

"""
    train_test_size_split(total_partition_size)
split DataFrame and return Array of DataFrame
"""
function train_test_size_split(total_partition_size)
    
    # train : valid : test = 0.64 : 0.16 : 0.20
    train_size = round(total_partition_size * 0.64)
    valid_size = round(total_partition_size * 0.16)
    test_size = total_partition_size - (train_size + valid_size)

    total_partition_size, Int(train_size), Int(valid_size), Int(test_size)
end

"""
    train_test_idx_split(tot_size, train_size, valid_size, test_size)
permute random indexes according to precomputed size
"""
function train_test_idxs_split(tot_size, train_size, valid_size, test_size)
    tot_idx = collect(range(1, stop=tot_size))
    
    tot_idx = Random.randperm(tot_size)
    train_idx = tot_idx[1: train_size]
    valid_idx = tot_idx[train_size + 1: train_size + valid_size]
    test_idx = tot_idx[train_size + valid_size + 1: end]

    sort!(train_idx), sort!(valid_idx), sort!(test_idx)
end

"""
    getHoursLater(df, hours, last_date_str, date_fmt)
get `hours` rows of DataFrame `df` after given `date`, String will be automatically  conveted to ZonedDateTime Object
"""
function getHoursLater(df::DataFrame, hours::Integer, last_date_str::String, date_fmt::Dates.DateFormat=Dates.DateFormat("yyyy-mm-dd HH:MM:SSz"))
    last_date = ZonedDateTime(last_date_str, date_fmt);

    getHoursLater(df, hours, last_date)
end

function getHoursLater(df::DataFrame, hours::Integer, last_date::ZonedDateTime)
    start_date = last_date + Hour(1)

    #date_range = collect(start_date:Hour(1):start_date+Hour(hours - 1))
    # collect within daterange
    df_hours = df |> @filter(_.date >= start_date &&
        _.date < (start_date + Hour(hours))) |> DataFrame

    df_hours
end

"""
    getX(df::DataFrame, idxs, features)
get X in Dataframe and construct X by flattening
"""
function getX(df::DataFrame, idxs, features::Array{Symbol})
    X = convert(Matrix, df[collect(idxs), features])

    vec(X)
end

getX(df::DataFrame, idxs, features::Array{String}) = getX(df, idxs, Symbol.(eval.(features)))

"""
    getY(X::DateFrame, hours)
get last date of X and construct Y with `hours` range
"""
function getY(X::DataFrame, ycol::String, hours=24)
    last_date_of_X = X[end, "date"]
    
    Y = get24HLater(X, ycol, last_date_of_X, hours)

    Y[:, Symbol(eval(ycol))]
end

"""
    make_minibatch(X::DataFrame, Y::DataFrame, idxs)
create minibatch
idx: partition by sample_size

"""
# Bundle images together with labels and group into minibatchess
function make_minibatch(df::Array{DataFrame}, ycol, idxs, features, hours)
    X_batch = getX(df, idxs, features)
    Y_batch = getY(X_batch, ycol, hours)
    #Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
