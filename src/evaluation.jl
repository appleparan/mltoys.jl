evaluations(dataset, model, μσ, metrics::Array{String}) =
    evaluations(dataset, model, μσ, Symbol.(metrics))

function evaluations(dataset, model, μσ, metrics::Array{Symbol})
    _μ = μσ["total", "μ"][:value]

    # initialize array per metric, i.e. RSR_arr
    for metric in metrics
        metric_arr = Symbol(metric, "_arr")
        eval(:($metric_arr = []))
    end

    # evaluate metric
    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        for metric in metrics
            # Expression of metric, i.e. RSR(y, ŷ, _μ)
            metric_func = :($(metric)($y, $ŷ, $_μ))
            # Expression of metric array, i.e. RSR_arr
            metric_arr = :($(Symbol(metric, "_arr")))

            # push to metric array, i.e. push!(RSR_arr, RSR(y, ŷ, _μ))
            push!(eval(metric_arr), eval(metric_func))
        end
    end

    # return metric arrays, i.e. (RSR_arr, NSR_arr, ...)
    Tuple(collect([eval(:(mean($(Symbol(metric, "_arr"))))) for metric in metrics]))
end

```
    RMSE(dataset, model, μσ)

RMSE-observations standard deviation ratio 
```
function RMSE(dataset, model, μσ::AbstractNDSparse)
    RMSE_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(RMSE_arr, RMSE(y, ŷ))
    end

    mean(RMSE_arr)
end

RMSE(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- ŷ)) / length(y))


```
    UNRMSE(dataset, model, μσ)

RMSE-observations standard deviation ratio, not normalized by length(y)
```
function UNRMSE(dataset, model, μσ::AbstractNDSparse)
    UNRMSE_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(UNRMSE_arr, UNRMSE(y, ŷ))
    end

    mean(UNRMSE_arr)
end

UNRMSE(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- ŷ)))

```
    RSR(dataset, model, μσ)

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

RMSE-observations standard deviation ratio

0 is best
```
function RSR(dataset, model, μσ::AbstractNDSparse)
    RSR_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    i = 0
    for (x, y) in dataset
        i += 1
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _RSR = RSR(y, ŷ, mean(y))

        if _RSR != Inf
            push!(RSR_arr, _RSR)
        end
    end

    mean(RSR_arr)
end

RSR(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- ŷ))) / sqrt(sum(abs2.(y .- μ)))

```
    NSE(dataset, model, μσ)

normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”, 1 is best
```
function NSE(dataset, model, μσ::AbstractNDSparse)
    NSE_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _NSE = NSE(y, ŷ, mean(y))

        if _NSE != Inf
            push!(NSE_arr, _NSE)
        end
    end

    mean(NSE_arr)
end

NSE(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sqrt(sum(abs2.(y .- ŷ))) / (sqrt(sum(abs2.(y .- μ))) + eps())

```
    PBIAS(dataset, model, μσ)
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
+ : underfitting
- : overfitting
```
function PBIAS(dataset, model, μσ::AbstractNDSparse)
    PBIAS_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(PBIAS_arr, PBIAS(y, ŷ))
    end

    mean(PBIAS_arr)
end

PBIAS(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(y .- ŷ) / (sum(y) + eps()) * 100.0

```
IOA(dataset, model, μσ)

Index of Agreement
The Index of Agreement (d) developed by Willmott (1981) 
as a standardized measure of the degree of model prediction error and varies between 0 and 1.

The index of agreement can detect additive and proportional differences in the observed and simulated means and variances;
however, it is overly sensitive to extreme values due to the squared differences (Legates and McCabe, 1999).

1 is best
```
function IOA(dataset, model, μσ::AbstractNDSparse)
    IOA_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        push!(IOA_arr, IOA(y, ŷ, mean(y)))
    end

    mean(IOA_arr)
end

IOA(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- ŷ)) /
        max(sum(abs.(ŷ .- μ) .+ abs.(y .- μ)), eps())

        
```
    R2(dataset, model, μσ)

    R-Squared

R2 assumes that every single variable explains the variation in the dependent variable. 
R Squared & Adjusted R Squared are often used for explanatory purposes and
explains how well your selected independent variable(s)
explain the variability in your dependent variable(s
https://en.wikipedia.org/wiki/Coefficient_of_determination
```
function R2(dataset, model, μσ::AbstractNDSparse)
    R2_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _R2 = R2(y, ŷ, mean(y))

        if _R2 != Inf
            push!(R2_arr, _R2)
        end
    end

    mean(R2_arr)
end

R2(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- ŷ)) / (sum(abs2.(y .- μ)))

```
    AdjR2(dataset, model, μσ::AbstractNDSparse)

the percentage of variation explained by only the independent variables that actually affect the dependent variable.
https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
```
function AdjR2(dataset, model, μσ::AbstractNDSparse)
    AdjR2_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _AdjR2 = AdjR2(y, ŷ, mean(y))

        if _AdjR2 != Inf
            push!(AdjR2_arr, _AdjR2)
        end
    end

    mean(AdjR2_arr)
end

function AdjR2(y, ŷ, μ::Real=zero(AbstractFloat)) 
    _R2 = R2(y, ŷ, μ)

    1.0 - (1.0 - _R2^2) * (length(y) - 1.0) / (length(y) - length(ŷ) - 1.0)
end

```
    RAE(dataset, model, μσ)

An absolute error(taking the absolute value of residual)
as a fraction of the actual value of the outcome target variable
```

function RAE(dataset, model, μσ::AbstractNDSparse)
    RAE_arr = Real[]
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
        @assert size(ŷ) == size(y)

        _RAE = RAE(y, ŷ, mean(y))

        if _RAE != Inf
            push!(RAE_arr, _RAE)
        end
    end

    mean(RAE_arr)
end

RAE(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(abs.(y .- ŷ)) / (sum(abs.(y .- μ)))

function classification(dataset::Array{T1, 1}, model, ycol::Symbol, μσ::AbstractNDSparse) where T1 <: Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}} where F <: AbstractFloat
    class_all_arr = []
    class_high_arr = []
    _μ = μσ["total", "μ"][:value]
    _σ = μσ["total", "σ"][:value]

    # TODO : if y is not 24 hour data, adjust them to use daily average
    for (x, y) in dataset
        ŷ = Flux.Tracker.data(model(x |> gpu))
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
