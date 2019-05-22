function RSME(dataset, model)
    # RSME
    acc = 0.0

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        acc += sqrt(sum(abs2.(ŷ .- y)) / length(y))
    end

    # acc is incomplete TrackedReal, convert it to pure Real type
    Flux.Tracker.data(acc / length(dataset))
end

evaluations(setname::String, dataset, model, μσ, metrics::Array{String}) =
    evaluations(setname, dataset, model, μσ, Symbol.(metrics))

function evaluations(setname::String, dataset, model, μσ, metrics::Array{Symbol})
    _μ = μσ[setname, "μ"][:value]

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
            # Expression of metric, i.e. RSR(y, ŷ, μ = _μ)
            metric_func = :($(metric)($y, $ŷ, $μ = _μ))
            # Expression of metric array, i.e. RSR_arr
            metric_arr = :($(Symbol(metric, "_arr")))

            # push to metric array, i.e. push!(RSR_arr, RSR(y, ŷ, μ = _μ))
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
function RMSE(setname::String, dataset, model, μσ)
    RMSE_arr = []

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(RSME_arr, RMSE(y, ŷ))
    end

    mean(RSME_arr)
end

RMSE(y, ŷ, μ=μ) = RMSE(y, ŷ)
RMSE(y, ŷ) = sqrt(sum(abs2.(y .- Flux.Tracker.data(cpu.(ŷ)))))

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
# MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

# RSR
#=
RMSE-observations standard deviation ratio 
=#
function RSR(setname::String, dataset, model, μσ)
    RSR_arr = []
    μ = μσ[setname, "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(RSR_arr, RSR(y, ŷ, μ=μ))
    end

    mean(RSR_arr)
end

RSR(y, ŷ; μ) = sqrt(sum(abs2.(y .- Flux.Tracker.data(cpu.(ŷ))))) / sqrt(sum(abs2.(y .- μ)))

# NSE
#=
normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”
=#
function NSE(setname::String, dataset, model, μσ)
    NSE_arr = []
    μ = μσ[setname, "μ"][:value]

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(NSE_arr, NSE(y, ŷ, μ=μ))
    end

    mean(NSE_arr)
end

NSE(y, ŷ; μ) = 1.0 - sqrt(sum(abs2.(y .- Flux.Tracker.data(cpu.(ŷ))))) / sqrt(sum(abs2.(y .- μ)))

#=
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
=#
function PBIAS(setname::String, dataset, model, μσ)
    PBIAS_arr = []

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        push!(PBIAS_arr, PBIAS(y, ŷ))
    end

    mean(PBIAS_arr)
end

PBIAS(y, ŷ, μ=μ) = PBIAS(y, ŷ)
PBIAS(y, ŷ) = sum(y .- Flux.Tracker.data(cpu.(ŷ))) / sum(y) * 100.0
