evaluations(dataset, model, statvals::AbstractNDSparse, features::Array{Symbol},
    metrics::Array{String}, f::Function=Flux.Tracker.data) =
    evaluations(dataset, model, statvals::AbstractNDSparse, features, Symbol.(metrics), f)

function evaluations(dataset, model, statvals::AbstractNDSparse,
    features::Array{Symbol}, metrics::Array{Symbol}, f::Function=Flux.Tracker.data)
    _μ = statvals["total", "μ"][:value]

    # initialize array per metric, i.e. RSR_arr
    metric_vals = []

    for metric in metrics
        if metric == :AdjR2
            # not implemented
            metric_func = :($(metric)($dataset, $model, $statvals, $(length)($feas), $f))
        else
            metric_func = :($(metric)($dataset, $model, $statvals, $f))
        end
        push!(metric_vals, eval(metric_func))
    end

    # unpack return values
    Tuple(metric_vals)
end

```
    RMSE(dataset, model, statvals, f = Flux.Tracker.data)

RMSE-observations standard deviation ratio 
```
function RMSE(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    RMSE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))

        @assert size(ŷ) == size(y)

        push!(RMSE_arr, RMSE(y, ŷ))
    end

    mean(RMSE_arr)
end

RMSE(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- ŷ)) / length(y))

```
    MAE(dataset, model, statvals)

Mean Absolute Error
```
function MAE(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    MAE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(MAE_arr, MAE(y, ŷ))
    end

    mean(MAE_arr)
end

MAE(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(abs.(y .- ŷ)) / length(y)

```
    MSPE(dataset, model, statvals::AbstractNDSparse)

Mean Square Percentage Error 

computed average of squared version of percentage errors 
by which forecasts of a model differ from actual values of the quantity being forecast.
https://en.wikipedia.org/wiki/Mean_percentage_error
```
function MSPE(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    MSPE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _MSPE = MSPE(y, ŷ, mean(y))

        if abs(_MSPE) != Inf
            #@assert 0 <= _MSPE <= 100.0
            push!(MSPE_arr, _MSPE)
        end
    end

    mean(MSPE_arr)
end

MSPE(y, ŷ, μ::Real=zero(AbstractFloat)) = 100.0 / length(y) * sum(abs2.((y .- ŷ) ./ y))

```
    MAPE(dataset, model, statvals::AbstractNDSparse)

Mean Absolute Percentage Error 

computed average of absolute version of percentage errors 
by which forecasts of a model differ from actual values of the quantity being forecast.
https://en.wikipedia.org/wiki/Mean_percentage_error
```
function MAPE(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    MAPE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _MAPE = MAPE(y, ŷ, mean(y))

        if abs(_MAPE) != Inf
            #@assert 0 <= _MAPE <= 100.0
            push!(MAPE_arr, _MAPE)
        end
    end

    mean(MAPE_arr)
end

MAPE(y, ŷ, μ::Real=zero(AbstractFloat)) = 100.0 / length(y) * sum(abs.((y .- ŷ) ./ y))

```
    RSR(dataset, model, statvals)

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

RMSE-observations standard deviation ratio

0 is best
```
function RSR(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    RSR_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    i = 0
    for (x, y) in dataset
        i += 1
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _RSR = RSR(y, ŷ, mean(y))

        if abs(_RSR) != Inf
            push!(RSR_arr, _RSR)
        end
    end

    mean(RSR_arr)
end

RSR(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- ŷ))) / sqrt(sum(abs2.(y .- μ)))

```
    NSE(dataset, model, statvals)

    Nash–Sutcliffe model efficiency coefficient

normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”.
1 is best, -∞ is worst
```
function NSE(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    NSE_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _NSE = NSE(y, ŷ, mean(y))

        if abs(_NSE) != Inf
            push!(NSE_arr, _NSE)
        end
    end

    mean(NSE_arr)
end

NSE(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sqrt(sum(abs2.(y .- ŷ))) / sqrt(sum(abs2.(y .- μ))) 

```
    PBIAS(dataset, model, statvals)
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
+ : underfitting
- : overfitting
```
function PBIAS(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    PBIAS_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(PBIAS_arr, PBIAS(y, ŷ))
    end

    mean(PBIAS_arr)
end

PBIAS(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(y .- ŷ) / (sum(y) + eps()) * 100.0

```
    IOA(dataset, model, statvals)

Index of Agreement

The Index of Agreement (d) developed by Willmott (1981) 
as a standardized measure of the degree of model prediction error and varies between 0 and 1.

The index of agreement can detect additive and proportional differences in the observed and simulated means and variances;
however, it is overly sensitive to extreme values due to the squared differences (Legates and McCabe, 1999).

1 is best, 0 is worst
```
function IOA(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    IOA_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _IOA = IOA(y, ŷ, mean(y))

        if abs(_IOA) != Inf
            if _IOA < 0 || _IOA > 1.0
                @show "Assertion Error: IOA, ", _IOA
            end
            @assert 0.0 <= _IOA <= 1.0

            push!(IOA_arr, _IOA)
        end
    end

    mean(IOA_arr)
end

IOA(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- ŷ)) /
        sum(abs2.(abs.(ŷ .- μ) .+ abs.(y .- μ)))

        ```
    RefinedIOA(dataset, model, statvals)

Refined Index of Agreement

1. reduce over-sensitive of IOA to large error-magnitude
Willmott,C. J., Robeson, S. M. and Matsuura, K. (2011).
A refined index of model performance. Int. J. Climatol. DOI: 10.1002/joc.2419
1 is best, 0 is worst
```
function RefinedIOA(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    RefinedIOA_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _RefinedIOA = RefinedIOA(y, ŷ, mean(y))

        if abs(_RefinedIOA) != Inf
            @assert -1.0 <= _RefinedIOA <= 1.0
            push!(RefinedIOA_arr, _RefinedIOA)
        end
    end

    mean(RefinedIOA_arr)
end

function RefinedIOA(y, ŷ, μ::Real=zero(AbstractFloat))
    c = 2
    m1 = sum(abs.(y .- ŷ))
    m2 = sum(abs.(y .- mean(y)))

    if m1 <= c * m2
        d = 1.0 - m1 / (c * m2)
    else
        d = c * m2 / m1 - 1.0
    end

    d
end

```
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
```
function R2(dataset, model, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data)
    R2_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _R2 = R2(y, ŷ, mean(y))

        if abs(_R2) != Inf
            @assert _R2 <= 1.0
            push!(R2_arr, _R2)
        end
    end

    mean(R2_arr)
end

R2(y, ŷ, μ::AbstractFloat) = 1.0 - sum(abs2.(y .- ŷ)) / sum(abs2.(y .- μ))

```
    AdjR2(dataset, model, statvals::AbstractNDSparse)

the percentage of variation explained by only the independent variables that actually affect the dependent variable.
https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
```
function AdjR2(dataset, model, statvals::AbstractNDSparse, k::Int=13, f::Function = Flux.Tracker.data)
    AdjR2_arr = Real[]
    _μ = statvals["total", "μ"][:value]
    _σ = statvals["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = f(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _AdjR2 = AdjR2(y, ŷ, mean(y), k)

        if abs(_AdjR2) != Inf
            push!(AdjR2_arr, _AdjR2)
        end
    end

    mean(AdjR2_arr)
end

function AdjR2(y, ŷ, μ::AbstractFloat, k::Int=13)
    _R2 = R2(y, ŷ, μ)

    1.0 - (1.0 - _R2^2) * (length(y) - 1.0) / (length(y) - k - 1.0)
end

function classification(dataset::Array{T1, 1}, model, ycol::Symbol, statvals::AbstractNDSparse, f::Function = Flux.Tracker.data) where {T1<:Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}} where F<:AbstractFloat}
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
