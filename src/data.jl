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
    sample_size::I, output_size::I, eltype::DataType=Float32) where I <: Integer

    # get X (2D)
    _X = eltype.(getX(df, features, sample_size))
    # get Y (1D)
    _Y = eltype.(getY(df, ycol, sample_size))

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
    eltype=Float32)

Create batch CNN Input for LSTNet with batch_size == 1
Used in valid/test set
"""
function make_pair_LSTNet(df::DataFrame,
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, kernel_length::I, output_size::I,
    eltype::DataType=Float32) where I <: Integer

    # O = (W - K + 2P) / S + 1
    # W = Input size, O = Output size
    # K = Kernel size, P = Padding size, S = Stride size
    pad_sample_size = kernel_length - 1
    X = zeros(eltype, pad_sample_size + sample_size, length(features), 1, 1)
    Y = zeros(eltype, output_size, 1)

    # get X (2D)
    _X = eltype.(getX(df, 1:sample_size, features))
    # get Y (1D)
    _Y = eltype.(getY(df, (sample_size + 1):(sample_size + output_size),
        ycol))

    # WHCN order, Channel is 1 because this is not an image
    # left zero padding
    X[(pad_sample_size + 1):end, :, 1, 1] = _X
    Y[:, 1] = _Y

    X = X |> gpu
    Y = Y |> gpu

    X, Y
end

"""
    make_batch_LSTNet(dfs, ycol, features,
    sample_size, kernel_length, output_size, batch_size,
    eltype=Float32)

Create batch CNN Input for LSTNet
Used in train set
"""
function make_batch_LSTNet(dfs::Array{DataFrame, 1},
    ycol::Symbol, features::Array{Symbol, 1},
    sample_size::I, kernel_length::I, output_size::I, batch_size::I,
    eltype::DataType=Float32) where I <: Integer

    # O = (W - K + 2P) / S + 1
    # W = Input size, O = Output size
    # K = Kernel size, P = Padding size, S = Stride size
    pad_sample_size = kernel_length - 1
    X = zeros(eltype, pad_sample_size + sample_size, length(features), 1, batch_size)
    Y = zeros(eltype, output_size, batch_size)
    
    # zero padding on input matrix
    for (i, df) in enumerate(dfs)
        # get X (2D)
        _X = eltype.(getX(df, 1:sample_size, features))
        # get Y (1D)
        _Y = eltype.(getY(df, (sample_size + 1):(sample_size + output_size),
            ycol))

        # WHCN order, Channel is 1 because this is not an image
        # left zero padding
        X[(pad_sample_size + 1):end, :, 1, i] = _X
        Y[:, i] = _Y
    end

    X = X |> gpu
    Y = Y |> gpu

    X, Y
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
    unpack_seq(X)

(sample_size, 1, hidCNN, batch_size)
-> [(hidCNN, batch_size),...]
"""
function unpack_seq(x::AbstractArray{N, 4}) where {N<:Number}
    @assert size(x, 2) == 1

    # (sample_size, 1, hidCNN, batch_size) ->
    # [(1, 1, (hidCNN, batch_size), 1), ...] (length: sample_size)

    x = permutedims(x, [3, 4, 1, 2])
    # TODO : disable scalar indexing on gpu
    #unpacked = [x[:, :, seq, 1] |> gpu for seq in axes(x, 3)]
    #unpacked

    mapslices(_x -> [_x], x, dims=[1, 2])
end

"""
    unpack_seq(X)

(sample_size, hidCNN, batch_size)
-> [(hidCNN, batch_size),...]
"""
function unpack_seq(x::AbstractArray{N, 3}) where {N<:Number}
    # (sample_size, hidCNN, batch_size)
    # -> (hidCNN, batch_size, sample_size)
    x = permutedims(x, [2, 3, 1])
    # TODO : disable scalar indexing on gpu
    #unpacked = [x[:, :, seq] |> gpu for seq in axes(x, 3)]
    #unpacked

    mapslices(_x -> [_x], x, dims=[1, 2])
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
    elseif method == :knn
        # TODO: KNN imputation is for multivarate.
        arr = collect(skipmissing(float.(df[!, ycol])))
        mat = collect(transpose(reshape(arr, size(arr, 1), 1)))
        kdtree = KDTree(mat)

        idxs, dists = knn(kdtree, mat, leafsize, true)
        w = ones(leafsize)
        for (i, _val) in enumerate(df[!, ycol])
            if _val === missing
                _idxs = idxs[i]
                _dists = dists[i]
                _w = w
                _imputed = sum(df[vcat(_idxs...), ycol] .* _w .* (1.0 .- _dists ./ sum(_dists))) / (sum(w) - 1)
                if isnan(_imputed) 
                    @show i, df[vcat(_idxs...), ycol]
                    @show sum(_dists), _dists
                end
                df[i, ycol] = Int(round())
            end
        end
    elseif method == :bpca
        # TODO: variational Bayes (VB) algorithm
        # bpca_init!(df, ycol, total_mean)
    end
end

#=
function bpca_fill(df::DataFrame, ycol_t::Symbol, ycols::Array{Symbol, 1})
    # init values
    # size of given array (N x D)
    N = size(df[!, ycol], 2)
    D = size(ycols, 1)
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
    # no miss data
    Rx = Iq .+ tau * transpose(W) * W + SigW;
    Rxinv = inv( Rx );
    dy = df[!, ycol] .- size(df[!, ycol], 2);
    x = tau * Rxinv * transpose(W) * transpose(dy);

    T = transpose(dy) * transpose(x);

    # trS
    sum( sum( dy .* dy )); 
end
=#