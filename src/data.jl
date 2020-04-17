function get_date_range(df::DataFrame, date_col_name::Symbol = :date)
    minimum(df[!, date_col_name]), maximum(df[!, date_col_name])
end

function filter_raw_data(df::DataFrame,
    test_fdate::D, test_tdate::D) where D<:Union{DateTime, ZonedDateTime}

    # filter dataframe by total date range and station Code
    df = @from i in df begin
        @where test_fdate <= i.date <= test_tdate
        @orderby i.date
        @select i
        @collect DataFrame
    end

    df
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
    filter_raw_data(df, stn_cdoe, fdate, tdate)

Used in analysis
"""

function filter_raw_data(df::DataFrame,
    stn_code::Integer,
    fdate::D, tdate::D) where D<:Union{DateTime, ZonedDateTime}

    # filter dataframe by total date range and station Code
    # @where fdate <= i.date <= tdate && i.stationCode == stn_code
    df = @from i in df begin
        @where i.stationCode == stn_code && fdate <= i.date <= tdate
        @orderby i.date
        @select i
        @collect DataFrame
    end

    df
end

function filter_station(df, stn_code)
    stn_df = @from i in df begin
        @where i.stationCode == stn_code
        @select i
        @collect DataFrame
    end

    stn_df
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
    @info "    Construct windows for current station..."

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
    sample_size::I, output_size::I, _eltype::DataType=Float32) where I <: Integer

    # get X (2D)
    _X = _eltype.(getX(df, features, sample_size))
    # get Y (1D)
    _Y = _eltype.(getY(df, ycol, sample_size))

    # for validation, sparse check not applied
    # GPU
    _X = vec(_X)
    _Y = _Y
    @assert length(_X) == sample_size * length(features)
    @assert length(_Y) == output_size
    @assert ndims(_X) == 1
    @assert ndims(_Y) == 1

    _X, _Y
end

"""
    make_pair_DNN(df, ycol, features, sample_size, output_size)

Split single window to serialized X and Y
"""
function make_pair_date_DNN(df::DataFrame,
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, output_size::I, _eltype::DataType=Float32) where I <: Integer

    # get X (2D)
    _X = _eltype.(getX(df, features, sample_size))
    # get Y (1D)
    _Y = _eltype.(getY(df, ycol, sample_size))
    _date = df[(sample_size+1):end, :date]

    # for validation, sparse check not applied
    # GPU
    _X = vec(_X)
    _Y = _Y
    @assert length(_X) == sample_size * length(features)
    @assert length(_Y) == output_size
    @assert ndims(_X) == 1
    @assert ndims(_Y) == 1

    _X, _Y, _date
end

"""
    make_minibatch_DNN(df, ycol, chnks, features, sample_size, output_size, batch_size, Float64)

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

https://discourse.julialang.org/t/flux-support-for-mini-batches/14718/3
"""
function make_batch_DNN(dfs::Array{DataFrame, 1}, ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, output_size::I, batch_size::I,
    missing_ratio::AbstractFloat=0.5, _eltype::DataType=Float32) where I <: Integer

    # check Dataframe array size be `batch_size`
    @assert length(dfs) <= batch_size

    X = zeros(_eltype, sample_size * length(features), batch_size)
    Y = zeros(_eltype, output_size, batch_size)

    # input_pairs Array{Tuple{Array{Int64,1},Array{Int64,1}},1}
    for (i, df) in enumerate(dfs)
        # get X (2D)
        _X = _eltype.(vec(getX(df, features, sample_size)))
        # get Y (1D)
        _Y = _eltype.(getY(df, ycol, sample_size))

        # fill zeros if data is too sparse (too many misisngs)
        # only in training
        remove_sparse_input!(_X, _Y)

        # make X & Y as column stacked batch
        X[:, i] = _X
        Y[:, i] = _Y
    end

    (X, Y)
end

"""
    getX(df::DataFrame, idxs, features, sample_size)

Get 2D input X from DataFrame.

return size: (sample_size, length(features))
"""
function getX(df::DataFrame, idxs::UnitRange{I}, features::Array{Symbol, 1}) where I <: Integer
    X = convert(Matrix, df[idxs, features])

    X
end

"""
    getY(df, ycol, sample_size, output_size)

Get 1D output Y from DataFrame.

return size: (output_size,)
"""
function getY(df::DataFrame, idxs::UnitRange{I}, ycol::Symbol) where I <: Integer
    Y = Array(df[idxs, ycol])

    Y
end

"""
    make_pair_LSTNet(df, ycol, features,
    sample_size, kernel_length, output_size,
    _eltype=Float32)

Create batch CNN Input for LSTNet with batch_size == 1
Used in valid/test set
"""
function make_pair_LSTNet(df::DataFrame,
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, kernel_length::I, output_size::I,
    _eltype::DataType=Float32) where I <: Integer

    # O = (W - K + 2P) / S + 1
    # W = Input size, O = Output size
    # K = Kernel size, P = Padding size, S = Stride size
    pad_sample_size = kernel_length - 1
    X_enc = zeros(_eltype, length(features), pad_sample_size + sample_size, 1, 1)
    # decode array (Array of Array, batch sequences)
    _x = zeros(_eltype, output_size, 1)
    X_dec = similar([_x], output_size)
    Y = zeros(_eltype, output_size, 1)

    # get X (2D)
    _X = _eltype.(getX(df, 1:sample_size, features))
    # get Y (1D)
    _Y = _eltype.(getY(df, (sample_size + 1):(sample_size + output_size),
        ycol))

    # WHCN order, Channel is 1 because this is not an image
    # left zero padding
    X_enc[:, (pad_sample_size + 1):end, 1, 1] = transpose(_X)
    Y[:, 1] = _Y

    # feed decoder output
    # Ref : Yujin Tang, et. al, Sequence-to-Sequence Model with Attention for Time Series Classification
    for i in 1:output_size
        # first row is zero, but from second row, it feeds output from previous time step
        # this makes to assert sample_size >= output_size
        _Y = _eltype.(getY(df, (i + 1):(i + output_size - 1), ycol))
        _X_dec = zeros(_eltype, output_size, 1)
        _X_dec[2:output_size] = _Y
        X_dec[i] = _X_dec
    end

    X_enc = X_enc |> gpu
    X_dec = X_dec |> gpu
    Y = Y |> gpu

    X_enc, X_dec, Y
end

"""
    make_batch_LSTNet(dfs, ycol, features,
    sample_size, kernel_length, output_size, batch_size,
    _eltype=Float32)

Create batch CNN Input for LSTNet
Used in train set
"""
function make_batch_LSTNet(dfs::Array{DataFrame, 1},
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, kernel_length::I, output_size::I, batch_size::I,
    _eltype::DataType=Float32) where I <: Integer

    # O = (W - K + 2P) / S + 1
    # W = Input size, O = Output size
    # K = Kernel size, P = Padding size, S = Stride size
    pad_sample_size = kernel_length - 1
    X_enc = zeros(_eltype, length(features), pad_sample_size + sample_size, 1, batch_size)
    # decode array (Array of Array, batch sequences)
    _x = zeros(_eltype, output_size, batch_size)
    X_dec = similar([_x], output_size)
    Y = zeros(_eltype, output_size, batch_size)

    # feed decoder output
    # zero padding on input matrix
    for (i, df) in enumerate(dfs)
        # get X (2D)
        _X = _eltype.(getX(df, 1:sample_size, features))
        # get Y (1D)
        _Y = _eltype.(getY(df, (sample_size + 1):(sample_size + output_size),
            ycol))

        # WHCN order, Channel is 1 because this is not an image
        # left zero padding
        X_enc[:, (pad_sample_size + 1):end, 1, i] = transpose(_X)
        Y[:, i] = _Y
    end

    # Ref : Yujin Tang, et. al, Sequence-to-Sequence Model with Attention for Time Series Classification
    for i in 1:output_size
        _X_dec = zeros(_eltype, output_size, batch_size)
        for (j, df) in enumerate(dfs)
            # first row is zero, but from second row, it feeds output from previous time step
            # this makes to assert sample_size >= output_size
            _Y = _eltype.(getY(df, (i + 1):(i + output_size - 1), ycol))

            _X_dec[2:output_size, j] = _Y
        end
        X_dec[i] = _X_dec
    end

    X_enc = X_enc |> gpu
    X_dec = X_dec |> gpu
    Y = Y |> gpu

    X_enc, X_dec, Y
end

"""
    serializeBatch(X, Y)

Reconstruct batch for RNN by serialize batch array.
Output will be arryas of 1D
"""
function serializeBatch(X, Y; dims=2)
    # after CNN, X and Y should have same size
    @assert size(X, 2) == size(Y, 2)

    collect(zip(eachslice(X, dims=dims), eachslice(Y, dims=dims)))
end

"""
    serializeBatch(X)

Convert 2D batch array to array of 1D array
(m x n) array -> n-of m x 1 arrays

i.e.
X = (m x n) -> [(m x 1), ...]
"""
serializeBatch(A; dims=2) = collect(eachslice(A, dims=dims))

"""
    whcn2cnh(X)

Convert 4D WHCN array to CNH

(1, sample_size, hidCNN, batch_size)
-> (hidCNN, batch_size, sample_size)
"""
function whcn2cnh(x::AbstractArray{N, 4}) where {N<:Number}
    @assert size(x, 1) == 1

    x = reshape(x, size(x, 2), size(x, 3), size(x, 4))

    permutedims(x, [2, 3, 1])
end

"""
    matrix2arrays(X)

2D matrix to array of arrays
"""
function matrix2arrays(x::AbstractArray{N, 2}) where {N<:Number}
    [x[i, :] |> gpu for i in axes(x, 1)]
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

zero2Missing!(df::DataFrame, ycol::Symbol) = replace!(df[!, ycol], 0 => missing, 0.0 => missing, NaN => missing)

"""
    impute!(df, ycol, method; total_mean)

# References
## BCPA
    * Shigeyuki Oba, et al. A BPCA Based Missing Value Imputing Method for Traffic Flow Volume Data
    * Li Qu, et. al. A Bayesian missing value estimation method for gene expression profile data
"""
#=
function impute!(df::DataFrame, ycol::Symbol, method::Symbol; total_mean::AbstractFloat = 0.0, ycols::Array{Symbol, 1} = [ycol], leafsize::Integer = 10)
    # filter missing value
    nomissings = filter(x -> x !== missing, df[!, ycol])

    if method == :mean
        replace!(df[!, ycol], missing => Int(round(total_mean)))
    elseif method == :sample
        _weeks = 53
        arr = collect(skipmissing(float.(df[!, ycol])))
        dfs = []

        for _week in 1:_weeks
            push!(dfs,
                DataFrames.filter(row -> week(row[:date]) == _week && row[ycol] !== missing && row[ycol] !== NaN, df))
        end

        weeks = week.(df[!, :date])
        for (i, _val) in enumerate(df[!, ycol])
            if _val === missing
                _df = dfs[weeks[i]]
                df[i, ycol] = sample(_df[!, ycol])
            end
        end
    #elseif method == :knn
    elseif method == :bpca
        # TODO: variational Bayes (VB) algorithm
        # bpca_init!(df, ycol, total_mean)
    end
end
=#

"""
    padded_push!(df, array, name, train_fdate, train_tdate)

append arrays to dataframe padded by train data

This use simple loop, so it may be slow
"""
padded_push!(df::DataFrame, array::AbstractArray, name::Symbol, train_fdate::ZonedDateTime, train_tdate::ZonedDateTime) =
    padded_push!(df, array, name, DateTime(train_fdate, Local), DateTime(train_tdate, Local))

function padded_push!(df::DataFrame, array::AbstractArray, name::Symbol, train_fdate::DateTime, train_tdate::DateTime)
    # get array filled with zero instead of missing
    # If I fill with missing, zscore fails due to msising value
    # for test_date, I don't need that.
    padded = zeros(size(df, 1))

    dates = train_fdate:Dates.Hour(1):train_tdate

    # should match
    @assert size(array) == size(dates)

    # now use dates as index by key-value structure
    train_values = Dict(dates .=> array)

    # fill value from date
    for (i, row) in enumerate(eachrow(df))
        date = DateTime(row[:date], Local)

        if haskey(train_values, date)
            padded[i] = train_values[date]
        end
    end

    DataFrames.insertcols!(df, 1, name => padded)
end

"""
    construct_annual_table(sea_year, sea_day, train_fdate, train_tdate)

Construct table of seasonality data within train date range
table accepts month, day, hour as keys.
"""
construct_annual_table(sea_year::AbstractVector, sea_day::AbstractVector, train_fdate::ZonedDateTime, train_tdate::ZonedDateTime) =
    construct_annual_table(sea_year, sea_day, DateTime(train_fdate, Local), DateTime(train_tdate, Local))

function construct_annual_table(df::DataFrame, target::Symbol, train_fdate::DateTime, train_tdate::DateTime)
    months = []
    days = []
    hours = []
    sea_years = []
    sea_days = []

    # ndsparse is immutable object, so make it dataframe first then group it.
    # 1. filter dataframe by date
    _df = DataFrames.filter(row -> train_fdate < DateTime(row[:date], Local) < train_tdate, df)

    # 2. Then make month, day, hour columns
    DataFrames.insertcols!(_df, 1, :month => Dates.month.(_df[!, :date]))
    DataFrames.insertcols!(_df, 1, :day => Dates.day.(_df[!, :date]))
    DataFrames.insertcols!(_df, 1, :hour => Dates.hour.(_df[!, :date]))

    # 2. group df using keys (month, day, hour)
    gd = DataFrames.groupby(_df, [:month, :day, :hour])

    # unique
    u_months = []
    u_days = []
    u_hours = []
    u_sea_years_fit = []
    u_sea_years_org = []
    u_sea_days = []

    for gdf in gd
        ugdf = unique(gdf)

        # length of ugdf is same as # of stations
        push!(u_months, ugdf[!, :month][1])
        push!(u_days, ugdf[!, :day][1])
        push!(u_hours, ugdf[!, :hour][1])
        push!(u_sea_years_fit, ugdf[!, Symbol(target, "_year_fit")][1])
        push!(u_sea_years_org, ugdf[!, Symbol(target, "_year_org")][1])
        push!(u_sea_days, ugdf[!, Symbol(target, "_day")][1])
    end

    ndsparse((month = u_months, day = u_days, hour = u_hours),
    (season_year_fit = u_sea_years_fit, season_year_org = u_sea_years_org,
    season_day = u_sea_days,))
end

"""
    padded_push!(df, array, name, train_fdate, train_tdate)

Append arrays to dataframe padded by test data
Assume that train data have been already appended,
so just replace missing to real value by given seaonality table

This use simple loop, so it may be slow
"""
padded_push!(df::DataFrame, target::Symbol,
    year_sea_col::Symbol, day_sea_col::Symbol, day_res_col::Symbol,
    season_table::AbstractNDSparse,
    test_fdate::ZonedDateTime, test_tdate::ZonedDateTime) =
    padded_push!(df, target, year_sea_col, day_sea_col, day_res_col, season_table,
    DateTime(test_fdate, Local), DateTime(test_tdate, Local))

function padded_push!(df::DataFrame, target::Symbol,
    year_sea_col::Symbol, day_sea_col::Symbol, day_res_col::Symbol,
    season_table::AbstractNDSparse,
    test_fdate::DateTime, test_tdate::DateTime)

    # fill value from table and compute residual
    for (i, row) in enumerate(eachrow(df))
        # check date in range
        if test_fdate <= DateTime(row.date, Local) <= test_tdate
            df[i, year_sea_col] = season_table[Dates.month(row.date), Dates.day(row.date), Dates.hour(row.date)].season_year_fit
            df[i, day_sea_col] = season_table[Dates.month(row.date), Dates.day(row.date), Dates.hour(row.date)].season_day
            df[i, day_res_col] = df[i, target] - df[i, year_sea_col] - df[i, day_sea_col]
        end
    end
end

padded_push!(df::DataFrame, target::Symbol,
    year_sea_fit_col::Symbol, year_sea_org_col::Symbol, day_sea_col::Symbol, day_res_col::Symbol,
    season_table::AbstractNDSparse,
    test_fdate::ZonedDateTime, test_tdate::ZonedDateTime) =
    padded_push!(df, target, year_sea_fit_col, year_sea_org_col, day_sea_col, day_res_col, season_table,
    DateTime(test_fdate, Local), DateTime(test_tdate, Local))

function padded_push!(df::DataFrame, target::Symbol,
    year_sea_fit_col::Symbol, year_sea_org_col::Symbol, day_sea_col::Symbol, day_res_col::Symbol,
    season_table::AbstractNDSparse,
    test_fdate::DateTime, test_tdate::DateTime)

    # fill value from table and compute residual
    for (i, row) in enumerate(eachrow(df))
        # check date in range
        if test_fdate <= DateTime(row.date, Local) <= test_tdate
            df[i, year_sea_fit_col] = season_table[Dates.month(row.date), Dates.day(row.date), Dates.hour(row.date)].season_year_fit
            df[i, year_sea_org_col] = season_table[Dates.month(row.date), Dates.day(row.date), Dates.hour(row.date)].season_year_org
            df[i, day_sea_col] = season_table[Dates.month(row.date), Dates.day(row.date), Dates.hour(row.date)].season_day
            df[i, day_res_col] = df[i, target] - df[i, year_sea_fit_col] - df[i, day_sea_col]
        end
    end
end

function compose_seasonality(dates::Array{DateTime, 1}, result, season_table::AbstractNDSparse, fitted=true)
    org_values = Array{Float64}(undef, size(dates, 1))

    if fitted
        for (i, date) in enumerate(dates)
            org_values[i] = result[i] +
                season_table[Dates.month(date), Dates.day(date), Dates.hour(date)].season_year_fit +
                season_table[Dates.month(date), Dates.day(date), Dates.hour(date)].season_day
        end
    else
        for (i, date) in enumerate(dates)
            org_values[i] = result[i] +
                season_table[Dates.month(date), Dates.day(date), Dates.hour(date)].season_year_org +
                season_table[Dates.month(date), Dates.day(date), Dates.hour(date)].season_day
        end
    end

    org_values
end
#=
function bpca_fill(df::DataFrame, ycol_t::Symbol, ycols::Array{Symbol, 1})
    # init values
    # size of given array (N x D)
    N = size(df[!, ycol], 2)
    # Univarate
    D = 1
    # default number of PCA axis
    Q = D - 1 > 0 ? D - 1 : D

    y_est_mean = df[!, ycol_t]
    y_est_zero = df[!, ycol_t]

    # for SVD,
    replace!(y_est_zero, missing => 0)
    # fill by mean values
    replace!(y_est_mean, missing => Int(round(total_mean)))

    mean_arr = map(ycols) do ycol
        mean(skipmissing(float.(df[!, ycol])))
    end

    # covariance
    covy = Statistics.cov(y_est_zero)
    # SVD
    F = LinearAlgebra.cov(cov)

    # PCA
    W = F.U * sqrt.(S)

    # variance
    tau = 1.0 ./( sum(Diagonal(covy)) - sum(Diagonal(F.S)) )

    taumax = 1e10;
    taumin = 1e-10;
    tau = max( min( tau, taumax), taumin );

    galpha0 = 1e-10;
    balpha0 = 1;
    alpha = ( 2 * galpha0 + d) ./ (tau * Diagonal(transpose(W)*W) + 2 * galpha0 / balpha0);

    gmu0  = 0.001;

    btau0 = 1;
    gtau0 = 1e-10;
    SigW = Matrix{Float64}(I, Q, Q);

    epochs = 200

    for i = 1:epochs

        trS = _bpca_no_miss()
    end
end

function _bpca_no_miss(Q::I), tau, W) where I<:Integer
    Iq = Matrix{Float64}(I, Q, Q)
    # E-step (Expectation)
    Rx = Iq .+ tau * transpose(W) * W + SigW;
    Rxinv = inv( Rx );
    dy = df[!, ycol] .- size(df[!, ycol], 2);
    x = tau * Rxinv * transpose(W) * transpose(dy);

    T = transpose(dy) * transpose(x);

    # trS
    sum( sum( dy .* dy ));

    # M-step (Maximization)
end
=#
