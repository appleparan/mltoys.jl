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
        ŷ = model(x |> gpu)
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

# RSR
#=
RMSE-observations standard deviation ratio 
=#
function RMSE(dataset, model, μσ::AbstractNDSparse)
    RMSE_arr = Real[]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(RMSE_arr, RMSE(y, ŷ))
    end

    mean(RMSE_arr)
end

RMSE(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- Flux.Tracker.data(ŷ))) / length(y))

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
# MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

# RSR
#=
RMSE-observations standard deviation ratio 
=#
function RSR(dataset, model, μσ::AbstractNDSparse)
    RSR_arr = Real[]
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(RSR_arr, RSR(y, ŷ, μ))
    end

    mean(RSR_arr)
end

RSR(y, ŷ, μ::Real=zero(AbstractFloat)) = sqrt(sum(abs2.(y .- Flux.Tracker.data(ŷ)))) / sqrt(sum(abs2.(y .- μ)))

# NSE
#=
normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”
=#
function NSE(dataset, model, μσ::AbstractNDSparse)
    NSE_arr = Real[]
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(NSE_arr, NSE(y, ŷ, μ))
    end

    mean(NSE_arr)
end

NSE(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sqrt(sum(abs2.(y .- Flux.Tracker.data(ŷ)))) / sqrt(sum(abs2.(y .- μ)))

#=
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
=#
function PBIAS(dataset, model, μσ::AbstractNDSparse)
    PBIAS_arr = Real[]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(PBIAS_arr, PBIAS(y, ŷ))
    end

    mean(PBIAS_arr)
end

PBIAS(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(y .- Flux.Tracker.data(ŷ)) / sum(y) * 100.0

function IOA(dataset, model, μσ::AbstractNDSparse)
    IOA_arr = Real[]
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(IOA_arr, IOA(y, ŷ, μ))
    end

    mean(IOA_arr)
end

IOA(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(Flux.Tracker.data(y) .- Flux.Tracker.data(ŷ))) /
        max(sum(abs.(Flux.Tracker.data(ŷ) .- μ) .+
            abs.(Flux.Tracker.data(y) .- μ)), eps())

# R2
#=
R-Squared
R Squared & Adjusted R Squared are often used for explanatory purposes and
explains how well your selected independent variable(s)
explain the variability in your dependent variable(s
https://en.wikipedia.org/wiki/Coefficient_of_determination
=#
function R2(dataset, model, μσ::AbstractNDSparse)
    R2_arr = Real[]
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(R2_arr, R2(y, ŷ, μ))
    end

    mean(R2_arr)
end

R2(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- Flux.Tracker.data(ŷ))) / sum(abs2.(y .- μ))

function RAE(Edataset, model, μσ::AbstractNDSparse)
    RAE_arr = Real[]
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(RAE_arr, RAE(y, ŷ, μ))
    end

    mean(RAE_arr)
end

RAE(y, ŷ, μ::Real=zero(AbstractFloat)) = sum(abs.(y .- Flux.Tracker.data(ŷ))) / sum(abs.(y .- μ))

#=
AdjR2 (TODO)
https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4

function AdjR2(dataset, model::F, μσ::AbstractNDSparse) where F <: Flux.Chain
    R2_arr = []
    μ = μσ["total", "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(AdjR2_arr, AdjR2(y, ŷ, μ))
    end

    mean(AdjR2_arr)
end

AdjR2(y, ŷ, μ::Real=zero(AbstractFloat)) = 1.0 - sum(abs2.(y .- Flux.Tracker.data(ŷ))) / sum(abs2.(y .- μ))
=#
function classification(dataset::Array{T1, 1}, model, ycol::Symbol) where T1 <: Tuple{AbstractArray{F, 1}, AbstractArray{F, 1}} where F <: AbstractFloat
    class_all_arr = []
    class_high_arr = []

    # TODO : if y is not 24 hour data, adjust them to use daily average
    for (x, y) in dataset
        ŷ = model(x |> gpu)
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
