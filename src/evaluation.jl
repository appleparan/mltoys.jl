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

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
# MODEL EVALUATION GUIDELINES FOR SYSTEMATIC QUANTIFICATION OF ACCURACY IN WATERSHED SIMULATIONS

# RSR
#=
RMSE-observations standard deviation ratio 
=#
function RSR(setname::String, dataset, model, μσ)
    RMSE = 0.0
    STD_obs = 0.0
    μ = μσ[setname, "μ"][:value]
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        RMSE += sum(abs2.(y .- ŷ))
        STD_obs += sum(abs2.(y .- μ))
    end

    sqrt(Flux.Tracker.data(RMSE)) / sqrt(Flux.Tracker.data(STD_obs))
end

# NSE
#=
normalized statistic that determines the
relative magnitude of the residual variance (“noise”)
compared to the measured data variance (“information”
=#
function NSE(setname::String, dataset, model, μσ)
    RMSE = 0.0
    STD_obs = 0.0
    μ = μσ[setname, "μ"][:value]
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        RMSE += sum(abs2.(y .- ŷ))
        STD_obs += sum(abs2.(y .- μ))
    end

    1.0 - Flux.Tracker.data(RMSE) / Flux.Tracker.data(STD_obs)
end

#=
The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. Positive values indicate model underestimation bias, and negative values
indicate model overestimation bias (Gupta et al., 1999)
=#
function PBIAS(setname::String, dataset, model, μσ)
    sim_tendency = 0.0
    obs = 0.0
    μ = μσ[setname, "μ"][:value]
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        sim_tendency += sum((y .- ŷ))
        obs += sum(y)
    end

    Flux.Tracker.data(sim_tendency) / Flux.Tracker.data(obs) * 100
end

