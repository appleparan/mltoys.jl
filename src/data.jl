function get_date_range(df::DataFrame, date_col_name::Symbol = :date)
    minimum(df[!, date_col_name]), maximum(df[!, date_col_name])
end

function filter_raw_data(df::DataFrame,
    train_fdate::D, train_tdate::D,
    test_fdate::D, test_tdate::D) where D<:Union{DateTime, ZonedDateTime}

    # filter dataframe by total date range and station Code
    df = @from i in df begin
        @where train_fdate <= i.date <= train_tdate ||
            test_fdate <= i.date <= test_tdate
        @orderby i.date
        @select i
        @collect DataFrame
    end

    df
end

"""
    validate_dates(_from_date, _to_date, window_size, df)

Validate dates range
"""
function validate_dates(_from_date::D, _to_date::D,
    window_size::Integer, df::DataFrame) where {I<:Integer, D<:Union{DateTime, ZonedDateTime}}
    # Round date and simple date range validation
    from_date = ceil(max(_from_date, df.date[1]), Dates.Hour(1))
    to_date = floor(min(_to_date, df.date[end]), Dates.Hour(1))
    if from_date > to_date
        throw(ArgumentError("invalid date range: $from_date ~ $to_date"))
    end

    # sample_size + output_size should be avalable
    # .value : to get `Dates` value
    if Dates.Hour(to_date - from_date).value < window_size
        throw(BoundsError(
            "window size($window_size) is smaller than date range: $from_date ~ $to_date"))
    end

    from_date, to_date
end

"""
    window_df(df, sample_size, output_size, offset = 0)

Create list of dataframe with sample_size + output_size window with offset
"""
function window_df(df::DataFrame, sample_size::I, output_size::I,
    offset::I = zero(I)) where I <: Integer

    # start index for window
    # sample_size + hours (hours for Y , < sample_size) should be avalable
    start_idxs = collect(1:(size(df, 1) - sample_size - output_size + 1))
    final_idxs = start_idxs .+ (sample_size + output_size - 1)
    
    # create Dataframe array
    dfs = DataFrame[]

    # filter dataframe by total date range and station Code
    for (si, fi) in zip(start_idxs, final_idxs)
        push!(dfs, df[(si+offset):(fi+offset), :])
    end

    dfs
end

"""
    window_df(df, sample_size, output_size, _from_date, _to_date)

Create list of dataframe with sample_size + output_size window with date range
"""
function window_df(df::DataFrame, sample_size::I, output_size::I,
    _from_date::D, _to_date::D) where {I<:Integer, D<:Union{DateTime, ZonedDateTime}}

    window_size = sample_size + output_size
    from_date, to_date = validate_dates(_from_date, _to_date, window_size, df)

    # create Dataframe array
    dfs = DataFrame[]

    total_it = Dates.Hour((to_date - Dates.Hour(window_size - 1)) - from_date).value
    p = Progress(total_it, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    @info "    Construct Training Set window for current station..."

    for _wd_from_date = from_date:Dates.Hour(1):(to_date - Dates.Hour(window_size - 1))
        ProgressMeter.next!(p);
        _wd_to_date = _wd_from_date + Dates.Hour(window_size)
        _df = @from i in df begin
            @where _wd_from_date <= i.date < _wd_to_date
            @orderby i.date
            @select i
            @collect DataFrame
        end
        
        push!(dfs, _df)
    end

    dfs
end

"""
    window_df(df, sample_size, output_size, _from_date, _to_date, station_code)

Create list of dataframe with sample_size + output_size window with date range and station code

TODO: Too slow!
"""
function window_df(df::DataFrame, sample_size::I, output_size::I,
    _from_date::D, _to_date::D,
    station_code::I) where {I<:Integer, D<:Union{DateTime, ZonedDateTime}}

    window_size = sample_size + output_size
    from_date, to_date = validate_dates(_from_date, _to_date, window_size, df)

    # create Dataframe array
    dfs = DataFrame[]

    total_it = Dates.Hour((to_date - Dates.Hour(window_size - 1)) - from_date).value
    p = Progress(total_it, dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    @info "    Construct Training Set window for station $(station_code)..."

    for _wd_from_date = from_date:Dates.Hour(1):(to_date - Dates.Hour(window_size - 1))
        ProgressMeter.next!(p);
        _wd_to_date = _wd_from_date + Dates.Hour(window_size)
        _df = @from i in df begin
            @where _wd_from_date <= i.date < _wd_to_date && i.stationCode == station_code
            @orderby i.date
            @select i
            @collect DataFrame
        end

        push!(dfs, _df)
    end
    
    dfs
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
    getX(df::DataFrame, features, sample_size)

Get 2D input X from DataFrame.

return size: (sample_size, length(features))
"""
function getX(df::DataFrame, features::Array{Symbol, 1}, sample_size::I) where I <: Integer
    X = convert(Matrix, df[1:sample_size, features])

    X
end

"""
    getY(df, ycol, sample_size, output_size)

Get 1D output Y from DataFrame.

return size: (output_size,)
"""
function getY(df::DataFrame, ycol::Symbol, sample_size::I) where I <: Integer
    Y = Array(df[(sample_size+1):end, ycol])

    Y
end

"""
    make_pair_DNN(df, ycol, features, sample_size, output_size)

Split single window to serialized X and Y
"""
function make_pair_DNN(df::DataFrame,
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, output_size::I, eltype::DataType=Float32) where I <: Integer

    # get X (2D)
    _X = eltype.(getX(df, features, sample_size))
    # get Y (1D)
    _Y = eltype.(getY(df, ycol, sample_size))

    # for validation, sparse check not applied
    # GPU
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

https://discourse.julialang.org/t/flux-support-for-mini-batches/14718/3?u=appleparan
"""
function make_batch_DNN(dfs::Array{DataFrame, 1}, ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, output_size::I, batch_size::I,
    missing_ratio::AbstractFloat=0.5, eltype::DataType=Float32) where I <: Integer

    # check Dataframe array size be `batch_size`
    @assert length(dfs) <= batch_size

    X = zeros(eltype, sample_size * length(features), batch_size)
    Y = zeros(eltype, output_size, batch_size)
    
    # input_pairs Array{Tuple{Array{Int64,1},Array{Int64,1}},1}
    for (i, df) in enumerate(dfs)
        # get X (2D)
        _X = eltype.(vec(getX(df, features, sample_size)))
        # get Y (1D)
        _Y = eltype.(getY(df, ycol, sample_size))

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
