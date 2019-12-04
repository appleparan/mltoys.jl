evaluations(dataset, model, statvals::AbstractNDSparse,
    metrics::Array{String}) =
    evaluations(dataset, model, statvals::AbstractNDSparse, Symbol.(metrics))
"""
    evaluations(dataset, model, statvals, metrics)
"""
function evaluations(dataset, model, statvals::AbstractNDSparse, metrics::Union{Array{Symbol}, Array{Function}})
    metric_vals = map(metrics) do metric
        evaluation(dataset, model, statvals, metric)
    end # do - end

    Tuple(metric_vals)
end

"""
    evaluation(dataset, model, statvals, metric)

compute evaluation metric (predefined) given by dataset
"""
function evaluation(dataset, model, statvals::AbstractNDSparse, metric::Symbol)
    eval(
    quote
        let _cnt = 0
            # column sum for batches
            _sum = sum($(dataset)) do xy
                _val = $(Symbol("_", metric))(xy[2], $(model)(xy[1]))

                # number of columns which is not Inf and not NaN
                _cnt += count(x -> !(isnan(x) || isinf(x)), _val)

                # If _val is Array -> use `replace!` to replace Inf, NaN to zero
                # If _val is Number -> just assign zero
                typeof(_val) <: AbstractArray ? replace!(_val, Inf => 0, -Inf => 0, NaN => 0) : ((isnan(_val) || isinf(_val)) && (_val = zero(_val)))

                _val
            end # do - end

            # mean
            _sum * 1 // _cnt
        end # let - end
    end) # quote - end
end

"""
    evaluation(dataset, model, statvals, metric)

compute evaluation metric (passed by Function or Anonymous Function) given by dataset
"""
function evaluation(dataset, model, statvals::AbstractNDSparse, metric)
    eval(quote
        let _cnt = 0
            # column sum for batches
            _sum = sum($(dataset)) do xy
                _val = $(metric)(xy[2], $(model)(xy[1]))

                # number of columns which is not Inf and not NaN
                _cnt += count(x -> !(isnan(x) || isinf(x)), _val)

                # If _val is Array -> use `replace!` to replace Inf, NaN to zero
                # If _val is Number -> just assign zero
                typeof(_val) <: AbstractArray ? replace!(_val, Inf => 0, -Inf => 0, NaN => 0) : ((isnan(_val) || isinf(_val)) && (_val = zero(_val)))

                _val
            end # do - end
            @show _sum
            # mean
            _sum * 1 // _cnt
        end # let - end
    end) # quote - end
end

# column sum
_RMSE(y::AbstractVector, ŷ::AbstractVector) = sqrt.(sum((y .- ŷ).^2, dims=[1]) * 1 // length(y))
_RMSE(y::AbstractMatrix, ŷ::AbstractMatrix) = sqrt.(sum((y .- ŷ).^2, dims=[1]) * 1 // size(y, 1))

"""
    RMSE(dataset, model, statvals, f = Flux.Tracker.data)

Root Mean Squared Error
"""
function RMSE(dataset, model, statvals::AbstractNDSparse)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        # column sum for batches
        _rmsesum = sum(dataset) do xy
            _rmseval = _RMSE(xy[2], model(xy[1]))

            # replace Inf, NaN to zero, no impact on sum
            replace!(_rmseval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _rmseval)

            _rmseval
        end

        # rmse mean
        _rmsesum / cnt
    end
end

# column sum
_MAE(y::AbstractVector, ŷ::AbstractVector) = sum(abs.(y .- ŷ), dims=[1]) * 1 // length(y)
_MAE(y::AbstractMatrix, ŷ::AbstractMatrix) = sum(abs.(y .- ŷ), dims=[1]) * 1 // size(y, 1)

"""
    MAE(dataset, model, statvals)

Compute Mean Absolute Error
"""
function MAE(dataset, model, statvals::AbstractNDSparse)
    MAE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _maesum = sum(dataset) do xy
            _maeval = _MAE(xy[2], model(xy[1]))

            # replace Inf, NaN to zero, no impact on sum
            replace!(_maeval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _maeval)

            _maeval
        end

        # mae mean
        _maesum / cnt
    end
end

# column sum
_MSPE(y::AbstractVector, ŷ::AbstractVector) = 100 // length(y) * sum(((y .- ŷ) ./ y).^2, dims=[1])
_MSPE(y::AbstractMatrix, ŷ::AbstractMatrix) = 100 // size(y, 1) * sum(((y .- ŷ) ./ y).^2, dims=[1])

"""
    MSPE(dataset, model, statvals::AbstractNDSparse)

Compute Mean Square Percentage Error

computed average of squared version of percentage errors 
by which forecasts of a model differ from actual values of the quantity being forecast.
https://en.wikipedia.org/wiki/Mean_percentage_error
"""
function MSPE(dataset, model, statvals::AbstractNDSparse)
    MSPE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _mspesum = sum(dataset) do xy
            _mspeval = _MSPE(xy[2], model(xy[1]))

            # replace Inf, NaN to zero, no impact on sum
            replace!(_mspeval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _mspeval)

            _mspeval
        end

        # mspe mean
        _mspesum / cnt
    end
end

# column sum
_MAPE(y::AbstractVector, ŷ::AbstractVector) = 100 // length(y) * sum(abs.((y .- ŷ) ./ y), dims=[1])
_MAPE(y::AbstractMatrix, ŷ::AbstractMatrix) = 100 // size(y, 1) * sum(abs.((y .- ŷ) ./ y), dims=[1])

"""
    MAPE(dataset, model, statvals::AbstractNDSparse)

Compute Mean Absolute Percentage Error

computed average of absolute version of percentage errors 
by which forecasts of a model differ from actual values of the quantity being forecast.
https://en.wikipedia.org/wiki/Mean_percentage_error
"""
function MAPE(dataset, model, statvals::AbstractNDSparse)
    MAPE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _mapesum = sum(dataset) do xy
            _mapeval = _MAPE(xy[2], model(xy[1]))

            # replace Inf, NaN to zero, no impact on sum
            replace!(_mapeval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _mapeval)

            _mapeval
        end

        # mape mean
        _mapesum / cnt
    end
end

_RSR(y::AbstractVecOrMat, ŷ::AbstractVecOrMat, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2, (y .- ŷ))) / sqrt(sum(abs2, (y .- μ)))

"""
    RSR(dataset, model, statvals)

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

RMSE-observations standard deviation ratio

0 is best
"""
function RSR(dataset, model, statvals::AbstractNDSparse)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _rsrsum = sum(dataset) do xy
            _rsrval = _RSR(xy[2], model(xy[1]), _μ)

            # replace Inf, NaN to zero, no impact on sum
            replace!(_rsrval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _rsrval)

            _rsrval
        end

        # RSR mean
        _rsrsum / cnt
    end
end

_NSE(y::AbstractVecOrMat, ŷ::AbstractVecOrMat, μ::Real=zero(AbstractFloat)) = 1.0 - sqrt(sum(abs2.(y .- ŷ))) / sqrt(sum(abs2.(y .- μ))) 

"""
    NSE(dataset, model, statvals)

    Nash–Sutcliffe model efficiency coefficient

normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”.
1 is best, -∞ is worst
"""
function NSE(dataset, model, statvals::AbstractNDSparse)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _nsesum = sum(dataset) do xy
            _nseval = _NSE(xy[2], model(xy[1]), _μ)

            # replace Inf, NaN to zero, no impact on sum
            replace!(_nseval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _nseval)

            _nseval
        end

        # NSE mean
        _nsesum / cnt
    end
end

_PBIAS(y::AbstractVecOrMat, ŷ::AbstractVecOrMat) = sum(y .- ŷ) / (sum(y) + eps()) * 100.0

"""
    PBIAS(dataset, model, statvals)
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
+ : underfitting
- : overfitting
"""
function PBIAS(dataset, model, statvals::AbstractNDSparse)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _pbiassum = sum(dataset) do xy
            _pbiasval = _PBIAS(xy[2], model(xy[1]))

            # replace Inf, NaN to zero, no impact on sum
            replace!(_pbiasval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _pbiasval)

            _pbiasval
        end

        # PBIAS mean
        _pbiassum / cnt
    end
end

_IOA(y::AbstractVecOrMat, ŷ::AbstractVecOrMat, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- ŷ)) /
        sum(abs2.(abs.(ŷ .- μ) .+ abs.(y .- μ)))

"""
    IOA(dataset, model, statvals)

Index of Agreement

The Index of Agreement (d) developed by Willmott (1981) 
as a standardized measure of the degree of model prediction error and varies between 0 and 1.

The index of agreement can detect additive and proportional differences in the observed and simulated means and variances;
however, it is overly sensitive to extreme values due to the squared differences (Legates and McCabe, 1999).

1 is best, 0 is worst
"""
function IOA(dataset, model, statvals::AbstractNDSparse)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _ioasum = sum(dataset) do xy
            _ioaval = _IOA(xy[2], model(xy[1]), _μ)

            # replace Inf, NaN to zero, no impact on sum
            replace!(_ioaval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _ioaval)

            _ioaval
        end
        # @assert 0.0 <= _IOA <= 1.0
        # IOA mean
        _ioasum / cnt
    end
end

function _RIOA(y::AbstractVecOrMat, ŷ::AbstractVecOrMat, μ::Real=zero(AbstractFloat))
    c = 2
    m1 = sum(abs.(y .- ŷ))
    m2 = sum(abs.(y .- μ))

    if m1 <= c * m2
        d = 1.0 - m1 / (c * m2)
    else
        d = c * m2 / m1 - 1.0
    end

    d
end

"""
    RefinedIOA(dataset, model, statvals)

Refined Index of Agreement

1. reduce over-sensitive of IOA to large error-magnitude
Willmott,C. J., Robeson, S. M. and Matsuura, K. (2011).
A refined index of model performance. Int. J. Climatol. DOI: 10.1002/joc.2419
1 is best, 0 is worst
"""
function RIOA(dataset, model, statvals::AbstractNDSparse, f::Function)    
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _rioasum = sum(dataset) do xy
            _rioaval = _RIOA(xy[2], model(xy[1]), _μ)

            # replace Inf, NaN to zero, no impact on sum
            replace!(_rioaval, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _rioaval)

            _rioaval
        end
        # @assert 0.0 <= _IOA <= 1.0
        # mean
        _rioasum / cnt
    end
end

_R2(y::AbstractVecOrMat, ŷ::AbstractVecOrMat, μ::AbstractFloat) = 1.0 - sum(abs2.(y .- ŷ)) / sum(abs2.(y .- μ))

"""
    R2(dataset, model, statvals)

    R-Squared

R2 assumes that every single variable explains the variation in the dependent variable. 
R Squared & Adjusted R Squared are often used for explanatory purposes and
explains how well your selected independent variable(s)
explain the variability in your dependent variable(s
https://en.wikipedia.org/wiki/Coefficient_of_determination

https://towardsdatascience.com/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-2-regression-metrics-d4a1a9ba3d74

Problems with R² that are corrected with an adjusted R²
1.  R² increases with every predictor added to a model.
    As R² always increases and never decreases,
    it can appear to be a better fit with the more terms you add to the model.
    This can be completely misleading.
2.  Similarly, if your model has too many terms and too many high-order polynomials
    you can run into the problem of over-fitting the data.
    When you over-fit data, a misleadingly high R² value can lead to misleading projections.
"""
function R2(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # sum(f, itr) + do block
    # no allocation + mapreduce => faster!
    let cnt = 0
        _r2sum = sum(dataset) do xy
            _r2val = _R2(xy[2], model(xy[1]), _μ)

            # replace Inf, NaN to zero, no impact on sum
            replace!(_r2val, Inf => 0, -Inf => 0, NaN => 0)
            # number of columns which is not Inf and not NaN
            cnt += count(x -> !(isnan(x) || isinf(x)), _r2val)

            _r2val
        end

        # R2 mean
        _r2sum / cnt
    end
end

function classification(dataset::Array{T1, 1}, model, ycol::Symbol, statvals::AbstractNDSparse) where {T1<:Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}} where F<:AbstractFloat}
    class_all_arr = []
    class_high_arr = []
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    # TODO : if y is not 24 hour data, adjust them to use daily average
    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        # daliy average
        mean_y = mean(y)
        mean_ŷ = mean(ŷ)

        # construct WHO function
        func_name = Symbol("WHO_", ycol)

        push!(class_all_arr, classification(mean_y, mean_ŷ, func_name))
        # push only high level
        if eval(:($(func_name)($mean_y))) > 2
            push!(class_high_arr, classification(mean_y, mean_ŷ, func_name))
        end
    end

    # length of zeros / array length * 100
    correct_all = count(x->x==0, class_all_arr) / length(class_all_arr)
    # length of zeros / array length * 100
    correct_high = count(x->x==0, class_high_arr) / length(class_high_arr)

    correct_all, correct_high
end

classification(mean_y, mean_ŷ, f) = abs(eval(:($(f)($mean_y))) - eval(:($(f)($mean_ŷ))))
